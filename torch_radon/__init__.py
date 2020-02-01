import numpy as np
import torch_radon_cuda
import torch
from torch import nn

from .differentiable_functions import RadonForward, RadonBackprojection
from .utils import compute_rays, normalize_shape


class Radon:
    def __init__(self, resolution, device=None):
        assert resolution % 2 == 0, "Resolution must be even"
        if device is None:
            device = torch.device('cuda')

        self.rays = compute_rays(resolution, device)

        # caches used to avoid reallocation of resources
        self.fp_tex_cache = torch_radon_cuda.TextureCache()
        self.bp_tex_cache = torch_radon_cuda.TextureCache()

        self.noise_generator = None

    @normalize_shape
    def forward(self, imgs, angles):
        return RadonForward.apply(imgs, self.rays, angles, self.fp_tex_cache, self.bp_tex_cache)

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
