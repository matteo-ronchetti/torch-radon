import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_radon import Radon
from utils import show_images

batch_size = 1
n_angles = 512
image_size = 512

img = np.load("phantom.npy")
device = torch.device('cuda')

# instantiate Radon transform
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles).to(device)

with torch.no_grad():
    x = torch.FloatTensor(img).reshape(1, 1, image_size, image_size).to(device)
    sinogram = radon.forward(x)
    filtered_sinogram = radon.filter_sinogram(sinogram)
    backprojection = radon.backprojection(sinogram, extend=True)
    fbp = radon.backprojection(filtered_sinogram, extend=False) * np.pi / n_angles

titles = ["Original Image", "Sinogram", "Filtered Sinogram", "Backprojection", "Filtered Backprojection"]
show_images([x, sinogram, filtered_sinogram, backprojection, fbp], titles, keep_range=False, shape=(2, 3))

plt.show()