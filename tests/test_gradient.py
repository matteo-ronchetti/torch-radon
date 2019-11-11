import torch
import numpy as np
from torch_radon import Radon
from unittest import TestCase


class TestGradient(TestCase):
    def test_differentiation(self):
        device = torch.device('cuda')
        x = torch.FloatTensor(1, 64, 64).to(device)
        x.requires_grad = True
        angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, 10).astype(np.float32)).to(device)

        radon = Radon(64).to(device)

        # check that backward is implemented for fp and bp
        y = radon.forward(x, angles)
        z = torch.mean(radon.backprojection(y, angles))
        z.backward()
