import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch_radon import Radon
# from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images, relative_error, circle_mask


def compare_images(imgs, titles=None, keep_range=True, shape=None, figsize=(8, 8.5)):
    combined_data = np.array(imgs)

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    # Get the min and max of all images
    if keep_range:
        _min, _max = np.amin(combined_data), np.amax(combined_data)
    else:
        _min, _max = None, None

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax[i].imshow(img, cmap=plt.cm.Greys_r, vmin=_min, vmax=_max)
        ax[i].set_title(title)


batch_size = 1
n_angles = 256
image_size = 256

img = cv2.imread("phantom.png", cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, 0]
img /= np.max(img)
img = cv2.resize(img, (256, 256))
# img = generate_random_images(1, 256)[0]
# print(img.shape)
# plt.imshow(img)
# plt.show()

device = torch.device('cuda')

# instantiate Radon transform and loss
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles).to(device)

# astra = AstraWrapper(angles)
# aid, astra_sinogram = astra.forward(img.reshape(1, image_size, image_size))
# astra_sinogram = astra_sinogram[0]
# astra_backprojection = astra.backproject(aid, image_size, 1)[0]  # * circle_mask(image_size)
# a_fbp = astra.fbp(img)
# plt.imshow(astra_backprojection[0])
# plt.show()

with torch.no_grad():
    x = torch.FloatTensor(img).reshape(1, 1, image_size, image_size).to(device)
    sinogram = radon.forward(x)
    readings = radon.emulate_readings(sinogram, 3.0, 12.0)
    filtered = radon.filter_sinogram(sinogram)
    y = radon.backprojection(sinogram, extend=True)
    fbp = radon.backprojection(filtered, extend=False) * np.pi / n_angles

# y = y[0, 0].cpu().numpy()  # * circle_mask(image_size)
# sinogram = sinogram[0, 0].cpu().numpy()

# print(relative_error(astra_sinogram, sinogram))
# compare_images([astra_sinogram, sinogram, astra_sinogram - sinogram], keep_range=False)
# print(relative_error(astra_backprojection, y))
# compare_images([astra_backprojection, y, astra_backprojection - y], keep_range=False)
# plt.show()
# compare_images([img, fbp[0, 0].cpu().numpy(), a_fbp])
plt.show()
# print(relative_error(astra_backprojection[0], y))
# diff = astra_backprojection[0] - y
# plt.imshow(diff)
# plt.show()
