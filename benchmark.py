from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images
import argparse
from examples.utils import show_images
import time
import numpy as np
import torch
from torch_radon import Radon
import torch_radon_cuda
import matplotlib.pyplot as plt

device = torch.device('cuda')


def bench(batch_size, image_size, n_angles=180, sample_size=50):
    x = generate_random_images(batch_size, image_size)
    angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)

    astra = AstraWrapper(angles)

    for i in range(10):
        _ = astra.forward(x)

    s = time.time()
    for i in range(sample_size):
        # for j in range(x.shape[0]):
        #     astra.fbp(x[j])
        _ = astra.forward(x)
    e = time.time()
    astra_time = (e - s) / sample_size

    with torch.no_grad():
        radon = Radon(image_size, angles).to(device)
        x = torch.FloatTensor(x)

        for i in range(10):
            x_ = x.to(device)
            y = radon.forward(x_)
            y = radon.filter_sinogram(y)
            z = radon.backprojection(y)
            y_ = z.cpu()

        s = time.time()
        for i in range(sample_size):
            x_ = x.to(device)
            y = radon.forward(x_)
            y = radon.filter_sinogram(y)
            z = radon.backprojection(y)
            y_ = z.cpu()

        e = time.time()
        our_time = (e - s) / sample_size

    return our_time, astra_time


def plot(tasks, astra_times, radon_time, radon_half_time, title):
    labels = tasks

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    params = {
        'text.usetex': True,
        'font.size': 8,
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, astra_times, width, label='Astra')
    rects2 = ax.bar(x, radon_time, width, label='Radon')
    rects3 = ax.bar(x + width, radon_half_time, width, label='Radon half precision')

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

    s = time.time()
    if sync:
        torch.cuda.synchronize()
    for _ in range(samples):
        f(x)
    if sync:
        torch.cuda.synchronize()
    e = time.time()

    return (e - s) / samples


def astra_forward_backward(astra, x, s, bs):
    pid, y = astra.forward(x)
    z = astra.backproject(pid, s, bs)
    astra.clean()
    return z


def radon_forward_backward(radon, x, half=False):
    if half:
        y = radon.forward(torch.HalfTensor(x).to(device))
        z = radon.backward(y.half())
        return z.cpu()

    y = radon.forward(torch.FloatTensor(x).to(device))
    z = radon.backward(y)
    return z


def main():
    parser = argparse.ArgumentParser(description='Benchmark and compare with Astra Toolbox')
    parser.add_argument('--task', default="all")
    parser.add_argument('--image-size', default=256, type=int)
    parser.add_argument('--angles', default=-1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--samples', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--output', default="")

    args = parser.parse_args()
    if args.angles == -1:
        args.angles = args.image_size

    device = torch.device("cuda")
    angles = np.linspace(0, 2 * np.pi, args.angles, endpoint=False).astype(np.float32)

    radon = Radon(args.image_size, angles)
    astra = AstraWrapper(angles)

    if args.task == "all":
        tasks = ["forward", "backward", "forward+backward", "forward from gpu"]
    else:
        tasks = [args.task]

    astra_fps = []
    radon_fps = []
    radon_half_fps = []

    # x = torch.randn((args.batch_size, args.image_size, args.image_size), device=device)

    if "forward" in tasks:
        print("Benchmarking forward")
        x = generate_random_images(args.batch_size, args.image_size)
        astra_time = benchmark_function(lambda y: astra.forward(y), x, args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon.forward(torch.FloatTensor(x).to(device)).cpu(), x, args.samples,
                                        args.warmup)
        radon_half_time = benchmark_function(lambda y: radon.forward(torch.HalfTensor(x).to(device)).cpu(), x,
                                             args.samples, args.warmup)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print(astra_time, radon_time, radon_half_time)
        astra.clean()

    if "backward" in tasks:
        print("Benchmarking backward")
        x = generate_random_images(args.batch_size, args.image_size)
        pid, x = astra.forward(x)

        astra_time = benchmark_function(lambda y: astra.backproject(pid, args.image_size, args.batch_size), x,
                                        args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon.backward(torch.FloatTensor(x).to(device)).cpu(), x,
                                        args.samples,
                                        args.warmup)
        radon_half_time = benchmark_function(lambda y: radon.backward(torch.HalfTensor(x).to(device)).cpu(), x,
                                             args.samples, args.warmup)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print(astra_time, radon_time, radon_half_time)
        astra.clean()

    if "forward+backward" in tasks:
        print("Benchmarking forward + backward")
        x = generate_random_images(args.batch_size, args.image_size)
        astra_time = benchmark_function(lambda y: astra_forward_backward(astra, y, args.image_size, args.batch_size), x,
                                        args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon_forward_backward(radon, y), x, args.samples,
                                        args.warmup)
        radon_half_time = benchmark_function(lambda y: radon_forward_backward(radon, y, half=True), x,
                                             args.samples, args.warmup)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print(astra_time, radon_time, radon_half_time)
        astra.clean()

    if "forward from gpu" in tasks:
        print("Benchmarking forward from device")
        x = generate_random_images(args.batch_size, args.image_size)
        dx = torch.FloatTensor(x).to(device)
        astra_time = benchmark_function(lambda y: astra.forward(y), x, args.samples, args.warmup)
        radon_time = benchmark_function(lambda y: radon.forward(y), dx, args.samples,
                                        args.warmup, sync=True)
        radon_half_time = benchmark_function(lambda y: radon.forward(y), dx.half(),
                                             args.samples, args.warmup, sync=True)

        astra_fps.append(args.batch_size / astra_time)
        radon_fps.append(args.batch_size / radon_time)
        radon_half_fps.append(args.batch_size / radon_half_time)

        print(astra_time, radon_time, radon_half_time)
        astra.clean()

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
