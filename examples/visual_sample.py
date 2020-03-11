import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from torch_radon import Radon
from torch_radon.solvers import Landweber
from utils import show_images

batch_size = 1
n_angles = 512
image_size = 512

img = np.load("phantom.npy")
device = torch.device('cuda')

# instantiate Radon transform
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles).to(device)

landweber = Landweber(radon)
alpha = landweber.estimate_alpha(image_size, device)
print(alpha, 2.0 / n_angles ** 2)

with torch.no_grad():
    x = torch.FloatTensor(img).reshape(1, 1, image_size, image_size).to(device)
    sinogram = radon.forward(x)
    filtered_sinogram = radon.filter_sinogram(sinogram)
    backprojection = radon.backprojection(sinogram, extend=True)
    fbp = radon.backprojection(filtered_sinogram, extend=False) * np.pi / n_angles
    land, progress = landweber.run(torch.zeros(x.size(), device=device), sinogram, alpha, iterations=500,
                                   callback=lambda xx: torch.norm(xx - x).item())

plt.plot(progress)

print("FBP Error", torch.norm(x - fbp).item())
print("Landweber Error", torch.norm(x - land).item())

titles = ["Original Image", "Sinogram", "Filtered Sinogram", "Backprojection", "Filtered Backprojection", "Landweber"]
show_images([x, sinogram, filtered_sinogram, backprojection, fbp, land], titles, keep_range=False, shape=(2, 3))

plt.show()
