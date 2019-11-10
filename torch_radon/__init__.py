import numpy as np
from torch import nn
from torch.autograd import Function
import torch

import torch_radon_cuda


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, rays, angles):
        sinogram = torch_radon_cuda.forward(x, rays, angles)
        ctx.save_for_backward(rays, angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.backward(grad_x, rays, angles), None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, rays, angles):
        image = torch_radon_cuda.backward(x, rays, angles)
        ctx.save_for_backward(rays, angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.forward(grad_x, rays, angles), None, None


class Radon(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        assert resolution % 2 == 0, "Resolution must be even"
        self.rays = nn.Parameter(self._compute_rays(resolution), requires_grad=False)

    def forward(self, imgs, angles):
        return RadonForward.apply(imgs, self.rays, angles)

    def backprojection(self, sinogram, angles):
        return RadonBackprojection.apply(sinogram, self.rays, angles)

    @staticmethod
    def _compute_rays(resolution):
        s = resolution // 2
        locations = np.arange(2 * s) - s + 0.5
        ys = np.sqrt(s ** 2 - locations ** 2)
        locations = locations.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        rays = np.hstack((locations, -ys, locations, ys))
        return torch.FloatTensor(rays)
