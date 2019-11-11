from tests.astra_wrapper import AstraWrapper
from tests.utils import generate_random_images, relative_error
import time
import numpy as np
import torch
from torch_radon import Radon

device = torch.device('cuda')

batch_size = 128
image_size = 128
n_angles = 128

x = generate_random_images(batch_size, image_size)
angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)

# astra = AstraWrapper(angles)
# a_id, a_y = astra.forward(x)

with torch.no_grad():
    radon = Radon(image_size).to(device)
    x = torch.FloatTensor(x)
    angles = torch.FloatTensor(angles).to(device)
    x_ = x.to(device)
    y = radon.forward(x_, angles) #.cpu().numpy()
    z = radon.backprojection(y, angles)
    
# print(relative_error(a_y, y))