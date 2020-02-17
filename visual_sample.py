import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch_radon import Radon

batch_size = 1
n_angles = 256
image_size = 256

img = cv2.imread("phantom.png", cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, 0]
img /= np.max(img)
img = cv2.resize(img, (256, 256))
print(img.shape)
plt.imshow(img)
plt.show()

device = torch.device('cuda')

# instantiate Radon transform and loss
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles).to(device)

with torch.no_grad():
    x = torch.FloatTensor(img).reshape(1, 1, image_size, image_size).to(device)
    sinogram = radon.forward(x)
    filtered = radon.filter_sinogram(sinogram)
    y = radon.backprojection(filtered)

print(sinogram.size())
plt.imshow(y[0, 0].cpu().numpy())
plt.show()
