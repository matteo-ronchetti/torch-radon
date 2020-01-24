import torch
import torch.nn as nn
import numpy as np
from torch_radon import Radon

batch_size = 16
n_angles = 256
image_size = 256

device = torch.device('cuda')

radon = Radon(image_size)
angles = torch.FloatTensor(np.linspace(0, np.pi, n_angles)).to(device)
x = torch.FloatTensor(batch_size, 8, image_size, image_size).to(device)
sinogram = radon.forward(x, angles)