from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images
import time
import numpy as np
import torch
from torch_radon import Radon

device = torch.device('cuda')


def bench(batch_size, image_size, n_angles=180, sample_size=50):
    x = generate_random_images(batch_size, image_size)
    angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)

    astra = AstraWrapper(angles)
    s = time.time()
    for i in range(sample_size):
        _ = astra.forward(x)
    e = time.time()
    astra_time = (e - s) / sample_size

    with torch.no_grad():
        radon = Radon(image_size, angles).to(device)
        x = torch.FloatTensor(x)

        s = time.time()
        for i in range(sample_size):
            x_ = x.to(device)
            y = radon.forward(x_)
            y_ = y.cpu()
        e = time.time()
        our_time = (e - s) / sample_size

    return our_time, astra_time


def main():
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

    print("BENCHMARKING FORWARD PROJECTION")
    for image_size in [64, 128, 256]:  # , 256, 512]:
        #for n_angles in [128]:
        n_angles = image_size
        print(f"Image size: {image_size}, Angles: {n_angles}")
        print("-------------------------------------------------------")
        for batch_size in [1, 8, 16, 32, 64]: #, 128, 256, 512]:
            our_time, astra_time = bench(batch_size, image_size, n_angles)
            print(f"Batch size: {batch_size}, Our: {batch_size / our_time} imgs/sec, Astra: {batch_size / astra_time} imgs/sec, speedup: {astra_time / our_time}")
        print("=======================================================")


main()
