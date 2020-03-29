import torch
import numpy as np
from torch_radon import Radon
from tests.utils import generate_random_images, relative_error
# import matplotlib.pyplot as plt
# import torch_radon_cuda

device = torch.device("cuda")

batch_size = 16
image_size = 128
angles = np.linspace(0, 2 * np.pi, 128).astype(np.float32)

# generate random images
# x = generate_random_images(batch_size, image_size)

# our implementation
radon = Radon(image_size, angles).to(device)
# x = torch.FloatTensor(x).to(device)
#
# sinogram = radon.forward(x)
# single_precision = radon.backprojection(sinogram, extend=True)
#
# half_precision = radon.backprojection(sinogram.half(), extend=True)
# print("Max", torch.max(half_precision))
#
# hp = half_precision.float().cpu().numpy()
#
# back_error = relative_error(single_precision.cpu().numpy(), hp)


x = torch.FloatTensor(16, 64, 64).to(device)
# lookup_table = torch.FloatTensor(128, 64).to(device)
# angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, 10).astype(np.float32))

radon = Radon(64, angles).to(device)

# sinogram = radon.forward(x)
# bp = radon.backprojection(sinogram)
readings = radon.emulate_readings(x, 5, 10.0)
# y = radon.readings_lookup(readings, lookup_table)


# plt.imshow(single_precision[0].cpu().numpy())
#
# plt.figure()
# plt.imshow(hp[0])
#
# plt.show()

# print(back_error)
# assert_less(back_error, 0)
