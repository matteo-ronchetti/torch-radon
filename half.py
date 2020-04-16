import torch
import numpy as np
from torch import nn
from torch_radon import Radon
from apex import amp
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        angles = np.linspace(0, 2 * np.pi, 64).astype(np.float32)
        self.radon = Radon(128, angles)
        self.conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        return self.conv(self.radon.forward(x))


device = torch.device("cuda")

batch_size = 16
image_size = 128

# generate random images
# x = generate_random_images(batch_size, image_size)

# our implementation
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters())

model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

x = torch.randn((1, 16, image_size, image_size)).to(device)

for i in range(15):
    sinogram = model.forward(x)

    print(sinogram[0,0].size())
    # plt.imshow(sinogram[0,0].float().detach().cpu())
    # plt.show()

    loss = torch.mean(sinogram)
    print(loss)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
