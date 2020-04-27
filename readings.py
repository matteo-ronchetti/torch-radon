from torch_radon import Radon, ReadingsLookup
import numpy as np
import torch

device = torch.device("cuda")
image_size = 256
angles = np.linspace(0, 2 * np.pi, 256).astype(np.float32)
radon = Radon(image_size, angles)

x = torch.abs(torch.randn((1, 32, image_size, image_size)).to(device)) / 256
sino = radon.forward(x)
lookup = ReadingsLookup(radon)

lookup.add_lookup_table(sino, 9.0, 5.0)