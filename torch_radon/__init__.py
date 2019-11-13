import numpy as np
from torch import nn
from torch.autograd import Function
import torch

import torch_radon_cuda


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache, back_tex_cache):
        sinogram = torch_radon_cuda.forward(x, rays, angles, tex_cache)
        ctx.back_tex_cache = back_tex_cache
        ctx.save_for_backward(rays, angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.backward(grad_x, rays, angles, ctx.back_tex_cache), None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache, back_tex_cache):
        image = torch_radon_cuda.backward(x, rays, angles, back_tex_cache)
        ctx.tex_cache = tex_cache
        ctx.save_for_backward(rays, angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.forward(grad_x, rays, angles, ctx.tex_cache), None, None, None, None


class Radon(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        assert resolution % 2 == 0, "Resolution must be even"
        self.rays = nn.Parameter(self._compute_rays(resolution), requires_grad=False)

        # caches used to avoid reallocation of resources
        self.fp_tex_cache = torch_radon_cuda.TextureCache()
        self.bp_tex_cache = torch_radon_cuda.TextureCache()

    def __del__(self):
        self.fp_tex_cache.free()
        self.bp_tex_cache.free()
    
    def forward(self, imgs, angles):
        return RadonForward.apply(imgs, self.rays, angles, self.fp_tex_cache, self.bp_tex_cache)

    def backprojection(self, sinogram, angles):
        return RadonBackprojection.apply(sinogram, self.rays, angles, self.fp_tex_cache, self.bp_tex_cache)

    @staticmethod
    def _compute_rays(resolution):
        s = resolution // 2
        locations = np.arange(2 * s) - s + 0.5
        ys = np.sqrt(s ** 2 - locations ** 2)
        locations = locations.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        rays = np.hstack((locations, -ys, locations, ys))
        return torch.FloatTensor(rays)
