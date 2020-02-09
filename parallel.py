import torch
from torch_radon import Radon
import torch.nn
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.radon = Radon(256)
        self.angles = torch.FloatTensor(np.linspace(0, np.pi, 256)).to(torch.device("cuda"))

    def forward(self, x):
        print(self.angles.device)
        return self.radon.forward(x, self.angles)


x = torch.FloatTensor(32, 1, 256, 256)  # .to(torch.device("cuda"))

model = Model()
model = torch.nn.DataParallel(model, [0, 1])
y = model(x)

#
