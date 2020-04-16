import numpy as np
import torch
from torch import nn
import scipy.stats

import torch_radon_cuda
from .differentiable_functions import RadonForward, RadonBackprojection
from .utils import compute_rays, normalize_shape


class Radon:
    def __init__(self, resolution, angles):
        super().__init__()

        assert resolution % 2 == 0, "Resolution must be even"
        self.resolution = resolution
        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        self.rays = compute_rays(resolution)  # nn.Parameter(compute_rays(resolution), requires_grad=False)
        self.angles = angles  # nn.Parameter(angles, requires_grad=False)

        # caches used to avoid reallocation of resources
        self.tex_cache = torch_radon_cuda.TextureCache(8)
        self.fft_cache = torch_radon_cuda.FFTCache(8)

        seed = np.random.get_state()[1][0]
        self.noise_generator = torch_radon_cuda.RadonNoiseGenerator(seed)

    def to(self, device):
        print("WARN Radon.to(device) is deprecated, device handling is now automatic")
        self._move_parameters_to_device(device)
        return self

    def _move_parameters_to_device(self, device):
        if device != self.rays.device:
            self.rays = self.rays.to(device)
            self.angles = self.angles.to(device)

    @normalize_shape
    def forward(self, imgs):
        assert imgs.size(-1) == self.resolution
        self._move_parameters_to_device(imgs.device)

        return RadonForward.apply(imgs, self.rays, self.angles, self.tex_cache)

    @normalize_shape
    def backprojection(self, sinogram, extend=False):
        assert sinogram.size(-1) == self.resolution
        self._move_parameters_to_device(sinogram.device)

        return RadonBackprojection.apply(sinogram, self.rays, self.angles, self.tex_cache, extend)

    @normalize_shape
    def backward(self, sinogram, extend=False):
        return self.backprojection(sinogram, extend)

    @normalize_shape
    def filter_sinogram(self, sinogram):
        return torch_radon_cuda.filter_sinogram(sinogram, self.fft_cache)

    @normalize_shape
    def add_noise(self, x, signal, density_normalization=1.0, approximate=False):
        print("WARN Radon.add_noise is deprecated")

        torch_radon_cuda.add_noise(x, self.noise_generator, signal, density_normalization, approximate)
        return x

    @normalize_shape
    def emulate_readings(self, x, signal, density_normalization=1.0):
        return torch_radon_cuda.emulate_sensor_readings(x, self.noise_generator, signal, density_normalization)

    @normalize_shape
    def readings_lookup(self, sensor_readings, lookup_table):
        return torch_radon_cuda.readings_lookup(sensor_readings, lookup_table)

    def set_seed(self, seed=-1):
        if seed < 0:
            seed = np.random.get_state()[1][0]

        self.noise_generator.set_seed(seed)

    def __del__(self):
        self.noise_generator.free()


def compute_lookup_table(sinogram, signal, normal_std, bins=4096, eps=0.01, eps_prob=0.99, verbose=False):
    s = sinogram.view(-1)
    device = s.device

    eps = np.quantile(sinogram.cpu().numpy(), 0.01)

    # Compute readings normalization value
    if verbose:
        print("Computing readings normalization value")
    k = 0
    for i in range(1, 20):
        a, b = torch_radon_cuda.compute_ab(s, signal, eps, bins * i)
        if verbose:
            print(a, b)
        if a >= (a + b) * eps_prob:
            k = bins * i
            break
    print("Readings normalization value = ", k)

    # Compute weights for Gaussian error
    norm_p = []
    norm_p_tot = 0
    for i in range(64):
        t = scipy.stats.norm.cdf((i + 0.5) / normal_std) - scipy.stats.norm.cdf((i - 0.5) / normal_std)
        norm_p.append(t)
        norm_p_tot += t
        if t / norm_p_tot < 0.05:
            break
    scale = k // 4096
    weights = [0.0] * (scale + 2 * len(norm_p) - 2)
    for i in range(len(norm_p)):
        for j in range(scale):
            weights[j - i + len(norm_p) - 1] += norm_p[i]
            if i > 0:
                weights[j + i + len(norm_p) - 1] += norm_p[i]

    # move weights to device
    weights = torch.FloatTensor(weights).to(device)

    lookup, lookup_var = torch_radon_cuda.compute_lookup_table(s, weights, signal, bins, k)

    return lookup, lookup_var, k
