import torch
import torch.nn as nn
import numpy as np
from torch_radon import Radon

batch_size = 16
n_angles = 256
image_size = 256

device = torch.device('cuda')

radon = Radon(image_size, np.linspace(0, np.pi, n_angles, endpoint=False)).to(device)

x = torch.FloatTensor(batch_size, 8, image_size, image_size).to(device)
sinogram = radon.forward(x)

for i in range(3):
    for batch_size in range(1, 8):
