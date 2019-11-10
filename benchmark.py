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
    astra_time = (e-s)/sample_size  
    #TODO clear astra
    
    
    radon = Radon(image_size).to(device)
    x = torch.FloatTensor(x).to(device)
    angles = torch.FloatTensor(angles).to(device)

    s = time.time()
    for i in range(sample_size):
        _ = radon.forward(x, angles)
    e = time.time()
    our_time = (e-s)/sample_size  
    
    return our_time, astra_time

for image_size in [64, 128]: #, 256, 512]:
    for n_angles in [180]:
        for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
            our_time, astra_time = bench(batch_size, image_size, n_angles)
            print(image_size, n_angles, batch_size, batch_size/our_time, batch_size/astra_time, astra_time/our_time)
        print("===========")