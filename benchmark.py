# from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images
import argparse
import astra
from examples.utils import show_images
import time
import numpy as np
import torch
from torch_radon import Radon, RadonFanbeam
import torch_radon_cuda
import matplotlib.pyplot as plt
from alpha_transform import AlphaShearletTransform
from torch_radon.shearlet import ShearletTransform

device = torch.device('cuda')


def plot(tasks, astra_times, radon_time, radon_half_time, title, lll=("Astra", "Radon", "Radon half precision")):
    labels = tasks

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    params = {
        # 'text.usetex': True,
        'font.size': 8,
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, astra_times, width, label=lll[0])
    rects2 = ax.bar(x, radon_time, width, label=lll[1])
    rects3 = ax.bar(x + width, radon_half_time, width, label=lll[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Images/second')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(np.round(height))}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()


def benchmark_function(f, x, samples, warmup, sync=False):
    for _ in range(warmup):
        f(x)

    if sync:
        torch.cuda.synchronize()
    s = time.time()
    for _ in range(samples):
        f(x)
    if sync:
        torch.cuda.synchronize()
    e = time.time()

    return (e - s) / samples


# def astra_forward_backward(astra, x, s, bs):
#     pid, y = astra.forward(x)
#     z = astra.backproject(pid, s, bs)
#     astra.clean()
#     return z
#
#
# def radon_forward_backward(radon, x, half=False):
#     if half:
#         y = radon.forward(torch.HalfTensor(x).to(device))
#         z = radon.backward(y.half())
#         return z.cpu()
#
#     y = radon.forward(torch.FloatTensor(x).to(device))
#     z = radon.backward(y)
#     return z.cpu()

class AstraParallelWrapper:
    def __init__(self, angles, img_size):
        self.angles = angles
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, img_size, self.angles)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

    def forward(self, x):
        device = x.device
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0], len(self.angles), x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            _, Y[i] = astra.create_sino(x[i], self.proj_id)

        return torch.from_numpy(Y).to(device)

    def backward(self, x):
        device = x.device
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            _, Y[i] = astra.create_backprojection(x[i], self.proj_id)

        return torch.from_numpy(Y).to(device)


class AstraFanbeamWrapper:
    def __init__(self, angles, img_size):
        self.angles = angles
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom('fanflat', 1.0, img_size, self.angles, img_size, img_size)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

    def forward(self, x):
        device = x.device
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0], len(self.angles), x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            _, Y[i] = astra.create_sino(x[i], self.proj_id)

        return torch.from_numpy(Y).to(device)

    def backward(self, x):
        device = x.device
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            _, Y[i] = astra.create_backprojection(x[i], self.proj_id)

        return torch.from_numpy(Y).to(device)


def shearlet_forward(alpha_shearlet, X):
    y = []
    for x in X:
        y.append(alpha_shearlet.transform(x, do_norm=False))
    return y


def benchmark_shearlet(args):
    img_size = args.image_size
    scales = [0.5] * 4

    x = np.random.uniform(0, 1, (args.batch_size, img_size, img_size))

    alpha_fps = []
    radon_fps = []
    radon_single_fps = []

    shearlet = ShearletTransform(img_size, img_size, scales)
    alpha_shearlet = AlphaShearletTransform(img_size, img_size, scales, real=True, parseval=True)

    coeff_ = alpha_shearlet.transform(x[0], do_norm=False)
    # rec_ = alpha_shearlet.adjoint_transform(coeff_, do_norm=False)

    alpha_forward_time = benchmark_function(lambda y: alpha_shearlet.transform(y, do_norm=False), x[0], args.samples,
                                            args.warmup)
    alpha_fps.append(1.0 / alpha_forward_time)
    print(alpha_forward_time)

    alpha_backward_time = benchmark_function(lambda y: alpha_shearlet.adjoint_transform(y, do_norm=False), coeff_, args.samples,
                                            args.warmup)
    alpha_fps.append(1.0 / alpha_backward_time)

    with torch.no_grad():
        dx = torch.DoubleTensor(x).to(device)
        radon_forward_time = benchmark_function(lambda y: shearlet.forward(y), dx, args.samples, args.warmup)
        radon_forward_fps = args.batch_size / radon_forward_time
        radon_fps.append(radon_forward_fps)

        coeff = shearlet.forward(dx)
        radon_backward_time = benchmark_function(lambda y: shearlet.backward(y), coeff, args.samples, args.warmup)
        radon_backward_fps = args.batch_size / radon_backward_time
        radon_fps.append(radon_backward_fps)

        dx = torch.FloatTensor(x).to(device)
        radon_forward_time = benchmark_function(lambda y: shearlet.forward(y), dx, args.samples, args.warmup)
        radon_forward_fps = args.batch_size / radon_forward_time
        radon_single_fps.append(radon_forward_fps)

        coeff = shearlet.forward(dx)
        radon_backward_time = benchmark_function(lambda y: shearlet.backward(y), coeff, args.samples, args.warmup)
        radon_backward_fps = args.batch_size / radon_backward_time
        radon_single_fps.append(radon_backward_fps)

    title = f"Image size {args.image_size}x{args.image_size}, batch size {args.batch_size} and scales {scales} on a {torch.cuda.get_device_name(0)}"

    plot(["Shearlet Forward", "Shearlet Back"], alpha_fps, radon_fps, radon_single_fps, title, ("Alpha Shearlet", "Torch Radon Double", "Torch Radon Float"))
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()
    # with torch.no_grad():
    #     # check with double precision
    #
    #     coeff = shearlet.forward(dx)
    #     rec = shearlet.backward(coeff)
    #
    #     x_err_d = (torch.norm(rec - dx) / torch.norm(dx)).item()
    #     coef_err_d = relative_error(coeff_, coeff.cpu().numpy())
    #     rec_err_d = relative_error(rec_, rec.cpu().numpy())
    #     print(x_err_d, coef_err_d, rec_err_d)
    #
    #     # check with single precision
    #     dx = torch.FloatTensor(x).to(device)
    #     coeff = shearlet.forward(dx)
    #     rec = shearlet.backward(coeff)


def main():
    parser = argparse.ArgumentParser(description='Benchmark and compare with Astra Toolbox')
    parser.add_argument('--task', default="all")
    parser.add_argument('--image-size', default=256, type=int)
    parser.add_argument('--angles', default=-1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--samples', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--output', default="")
    parser.add_argument('--circle', action='store_true')

    args = parser.parse_args()
    if args.angles == -1:
        args.angles = args.image_size

    device = torch.device("cuda")
    angles = np.linspace(0, 2 * np.pi, args.angles, endpoint=False).astype(np.float32)

    radon = Radon(args.image_size, angles, clip_to_circle=args.circle)
    radon_fb = RadonFanbeam(args.image_size, angles, args.image_size, clip_to_circle=args.circle)

    astra_pw = AstraParallelWrapper(angles, args.image_size)
    astra_fw = AstraFanbeamWrapper(angles, args.image_size)
    # astra = AstraWrapper(angles)

    if args.task == "all":
        tasks = ["forward", "backward", "fanbeam forward", "fanbeam backward"]
    elif args.task == "shearlet":
        # tasks = ["shearlet forward", "shearlet backward"]
        benchmark_shearlet(args)
        return
    else:
        tasks = [args.task]

    astra_fps = []
    radon_fps = []
    radon_half_fps = []

    if "forward" in tasks:
        print("Benchmarking forward from device")
        x = generate_random_images(args.batch_size, args.image_size)
        dx = torch.FloatTensor(x).to(device)

        astra_time = benchmark_function(lambda y: astra_pw.forward(y), dx, args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon.forward(y), dx, args.samples,
                                        args.warmup, sync=True)
        radon_half_time = benchmark_function(lambda y: radon.forward(y), dx.half(),
                                             args.samples, args.warmup, sync=True)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print("Speedup:", astra_time / radon_time)
        print("Speedup half-precision:", astra_time / radon_half_time)
        print()

    if "backward" in tasks:
        print("Benchmarking backward from device")
        x = generate_random_images(args.batch_size, args.image_size)
        dx = torch.FloatTensor(x).to(device)

        astra_time = benchmark_function(lambda y: astra_pw.backward(y), dx,
                                        args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon.backward(y), dx, args.samples,
                                        args.warmup, sync=True)
        radon_half_time = benchmark_function(lambda y: radon.backward(y), dx.half(),
                                             args.samples, args.warmup, sync=True)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print("Speedup:", astra_time / radon_time)
        print("Speedup half-precision:", astra_time / radon_half_time)
        print()

    if "fanbeam forward" in tasks:
        print("Benchmarking fanbeam forward")
        x = generate_random_images(args.batch_size, args.image_size)
        dx = torch.FloatTensor(x).to(device)
        #
        astra_time = benchmark_function(lambda y: astra_fw.forward(y), dx,
                                        args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon_fb.forward(y), dx, args.samples,
                                        args.warmup, sync=True)
        radon_half_time = benchmark_function(lambda y: radon_fb.forward(y), dx.half(),
                                             args.samples, args.warmup, sync=True)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print("Speedup:", astra_time / radon_time)
        print("Speedup half-precision:", astra_time / radon_half_time)
        print()

    if "fanbeam backward" in tasks:
        print("Benchmarking fanbeam backward")
        x = generate_random_images(args.batch_size, args.image_size)
        dx = torch.FloatTensor(x).to(device)
        #
        astra_time = benchmark_function(lambda y: astra_fw.backward(y), dx,
                                        args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon_fb.backprojection(y), dx, args.samples,
                                        args.warmup, sync=True)
        radon_half_time = benchmark_function(lambda y: radon_fb.backprojection(y), dx.half(),
                                             args.samples, args.warmup, sync=True)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print("Speedup:", astra_time / radon_time)
        print("Speedup half-precision:", astra_time / radon_half_time)
        print()

    title = f"Image size {args.image_size}x{args.image_size}, {args.angles} angles and batch size {args.batch_size} on a {torch.cuda.get_device_name(0)}"

    plot(tasks, astra_fps, radon_fps, radon_half_fps, title)
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()


# device = torch.device("cuda")
# image_size = 256
# batch_size = 32 * 16
# angles = np.linspace(0, 2 * np.pi, 256).astype(np.float32)
# radon = Radon(image_size, angles).to(device)
#
# x = torch.randn((batch_size, image_size, image_size), device=device)
#
# sino = torch_radon_cuda.forward(x, radon.rays, radon.angles, radon.tex_cache)
#
# with torch.no_grad():
#     torch.cuda.synchronize()
#     s = time.time()
#     for i in range(25):
#         y = radon.forward(x)  # torch_radon_cuda.forward(sino, radon.rays, radon.angles, radon.tex_cache, True)
#     torch.cuda.synchronize()
#     e = time.time()
#     print(e - s)
#
#     ss = sino.half()
#     xx = x.half()
#     torch.cuda.synchronize()
#     s = time.time()
#     for i in range(25):
#         y = radon.forward(xx)  # torch_radon_cuda.backward(ss, radon.rays, radon.angles, radon.tex_cache, True)
#     torch.cuda.synchronize()
#     print(y.size())
#     e = time.time()
#     print(e - s)


#
# with torch.no_grad():
#
# device = torch.device("cuda")
# batch_size = 256
# image_size = 256
# sample_size = 50
# angles = np.linspace(0, 2 * np.pi, 256).astype(np.float32)
#
# with torch.no_grad():
#     radon = Radon(image_size, device)
# x = generate_random_images(batch_size, image_size)
# x = torch.FloatTensor(x).to(device)
# angles = torch.FloatTensor(angles).to(device)
#
# torch.cuda.synchronize()
# s = time.time()
# for i in range(sample_size):
#     y = radon.forward(x, angles)
# torch.cuda.synchronize()
# e = time.time()
# our_time = (e - s) / sample_size
# print(our_time, batch_size/our_time)
#
# torch.cuda.synchronize()
# s = time.time()
# for i in range(sample_size):
#     z = radon.backprojection(y, angles)
# torch.cuda.synchronize()
# e = time.time()
# our_time = (e - s) / sample_size
# print(our_time, batch_size/our_time)

# print("BENCHMARKING FORWARD PROJECTION")
# for image_size in [64, 128, 256]:  # , 256, 512]:
#     #for n_angles in [128]:
#     n_angles = image_size
#     print(f"Image size: {image_size}, Angles: {n_angles}")
#     print("-------------------------------------------------------")
#     for batch_size in [1, 8, 16, 32, 64]: #, 128, 256, 512]:
#         our_time, astra_time = bench(batch_size, image_size, n_angles)
#         print(f"Batch size: {batch_size}, Our: {batch_size / our_time} imgs/sec, Astra: {batch_size / astra_time} imgs/sec, speedup: {astra_time / our_time}")
#     print("=======================================================")


main()
