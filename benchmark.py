from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images
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


def main():
    device = torch.device("cuda")
    image_size = 256
    batch_size = 32 * 16
    angles = np.linspace(0, 2 * np.pi, 256).astype(np.float32)
    radon = Radon(image_size, angles).to(device)

    x = torch.randn((batch_size, image_size, image_size), device=device)

    sino = torch_radon_cuda.forward(x, radon.rays, radon.angles, radon.tex_cache)

    y = torch_radon_cuda.backward(sino, radon.rays, radon.angles, radon.tex_cache, True)
    ss = sino.permute(1, 2, 0).contiguous()
    y_ = torch_radon_cuda.backward_lb(ss, radon.rays, radon.angles, radon.tex_cache, True)

    print(torch.norm(y - y_) / torch.norm(y))
    # show_images([y[0], y_[0]])
    # plt.show()

    # print(sino.size())
    #
    # with torch.no_grad():
    #     torch.cuda.synchronize()
    #     s = time.time()
    #     for i in range(10):
    #         y = torch_radon_cuda.backward(sino, radon.rays, radon.angles, radon.tex_cache, True)
    #     torch.cuda.synchronize()
    #     e = time.time()
    #     print(e - s)
    #
    #     torch.cuda.synchronize()
    #     s = time.time()
    #     for i in range(10):
    #         ss = sino.permute(1, 2, 0).contiguous()
    #         y = torch_radon_cuda.backward_lb(ss, radon.rays, radon.angles, radon.tex_cache, True)
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
