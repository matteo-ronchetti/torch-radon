import numpy as np
import torch
from torch import nn

import torch_radon_cuda
from .differentiable_functions import RadonForward, RadonBackprojection
from .utils import compute_rays, normalize_shape


class Radon(nn.Module):
    def __init__(self, resolution, angles):
        super().__init__()

        assert resolution % 2 == 0, "Resolution must be even"
        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        self.rays = nn.Parameter(compute_rays(resolution), requires_grad=False)
        self.angles = nn.Parameter(angles, requires_grad=False)

        # caches used to avoid reallocation of resources
        self.fp_tex_cache = torch_radon_cuda.TextureCache()
        self.bp_tex_cache = torch_radon_cuda.TextureCache()

        self.noise_generator = None

    @normalize_shape
    def forward(self, imgs):
        # print("Rays", self.rays.device)
        # return imgs
        return RadonForward.apply(imgs, self.rays, self.angles, self.fp_tex_cache, self.bp_tex_cache)

    @normalize_shape
    def backprojection(self, sinogram, angles):
        return RadonBackprojection.apply(sinogram, self.rays, angles, self.fp_tex_cache, self.bp_tex_cache)

    @normalize_shape
    def filter_sinogram(self, sinogram):
        return torch_radon_cuda.filter_sinogram(sinogram)

    @normalize_shape
    def add_noise(self, x, signal, density_normalization=1.0, approximate=False):
        if self.noise_generator is None:
            self.set_seed()

        torch_radon_cuda.add_noise(x, self.noise_generator, signal, density_normalization, approximate)
        return x

    @normalize_shape
    def emulate_readings(self, x, signal, density_normalization=1.0):
        if self.noise_generator is None:
            self.set_seed()

        return torch_radon_cuda.emulate_sensor_readings(x, self.noise_generator, signal, density_normalization)

    @normalize_shape
    def readings_lookup(self, sensor_readings, lookup_table):
        return torch_radon_cuda.readings_lookup(sensor_readings, lookup_table)

    def set_seed(self, seed=-1):
        if seed < 0:
            seed = np.random.get_state()[1][0]

        if self.noise_generator is not None:
            self.noise_generator.free()

        self.noise_generator = torch_radon_cuda.RadonNoiseGenerator(seed)

    def __del__(self):
        self.fp_tex_cache.free()
        self.bp_tex_cache.free()

        if self.noise_generator is not None:
            self.noise_generator.free()
