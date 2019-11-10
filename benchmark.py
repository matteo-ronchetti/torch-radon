from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images
import time
import numpy as np
import torch
from torch_radon import Radon

device = torch.device('cuda')

def bench(batch_size, image_size, n_angles=180):
    x = generate_random_images(batch_size, image_size)
    angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)
    radon = Radon(image_size).to(device)
    x = torch.FloatTensor(x).to(device)
    angles = torch.FloatTensor(angles).to(device)

    s = time.time()
    for i in range(50):
        our_fp = radon.forward(x, angles)
    e = time.time()
    
    return (e-s)/50

for image_size in [64, 128, 256, 512]:
    for n_angles in [180]:
        for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
            print(image_size, n_angles, batch_size, batch_size/bench(batch_size, image_size, n_angles))
        print("===========")