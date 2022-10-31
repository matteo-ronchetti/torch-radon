import numpy as np
import torch
import scipy.stats
import abc
import torch.nn.functional as F
import warnings

try:
    import torch_radon_cuda
    from torch_radon_cuda import RaysCfg
except Exception as e:
    print("Importing exception")

from .differentiable_functions import RadonForward, RadonBackprojection
from .utils import normalize_shape
from .filtering import FourierFilters

__version__ = "1.0.0"


class BaseRadon(abc.ABC):
    def __init__(self, angles, rays_cfg):
        self.rays_cfg = rays_cfg

        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        # change sign to conform to Astra and Scikit
        self.angles = -angles

        # caches used to avoid reallocation of resources
        self.tex_cache = torch_radon_cuda.TextureCache(8)
        self.fourier_filters = FourierFilters()

        seed = np.random.get_state()[1][0]
        self.noise_generator = torch_radon_cuda.RadonNoiseGenerator(seed)

    def _move_parameters_to_device(self, device):
        if device != self.angles.device:
            self.angles = self.angles.to(device)

    def _check_input(self, x, square=False):
        if not x.is_contiguous():
            x = x.contiguous()

        if square:
            assert x.size(1) == x.size(2), f"Input images must be square, got shape ({x.size(1)}, {x.size(2)})."

        if x.dtype == torch.float16:
            assert x.size(
                0) % 4 == 0, f"Batch size must be multiple of 4 when using half precision. Got batch size {x.size(0)}"

        return x

    @normalize_shape(2)
    def forward(self, x):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        """
        x = self._check_input(x, square=True)
        self._move_parameters_to_device(x.device)

        return RadonForward.apply(x, self.angles, self.tex_cache, self.rays_cfg)

    @normalize_shape(2)
    def backprojection(self, sinogram):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        """
        sinogram = self._check_input(sinogram)
        self._move_parameters_to_device(sinogram.device)

        return RadonBackprojection.apply(sinogram, self.angles, self.tex_cache, self.rays_cfg)

    @normalize_shape(2)
    def filter_sinogram(self, sinogram, filter_name="ramp"):
        # if not self.clip_to_circle:
        #     warnings.warn("Filtered Backprojection with clip_to_circle=True will not produce optimal results."
        #                   "To avoid this specify clip_to_circle=False inside Radon constructor.")

        # Pad sinogram to improve accuracy
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size

        padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))
        # TODO should be possible to use onesided=True saving memory and time
        sino_fft = torch.fft.fft(padded_sinogram)

        # get filter and apply
        f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
        filtered_sino_fft = sino_fft * f.squeeze(2).unsqueeze(1)

        # Inverse fft
        filtered_sinogram = torch.real(torch.fft.ifft(filtered_sino_fft))

        # pad removal and rescaling
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)

    def backward(self, sinogram):
        r"""Same as backprojection"""
        return self.backprojection(sinogram)

    @normalize_shape(2)
    def add_noise(self, x, signal, density_normalization=1.0, approximate=False):
        # print("WARN Radon.add_noise is deprecated")

        torch_radon_cuda.add_noise(x, self.noise_generator, signal, density_normalization, approximate)
        return x

    @normalize_shape(2)
    def emulate_readings(self, x, signal, density_normalization=1.0):
        return torch_radon_cuda.emulate_sensor_readings(x, self.noise_generator, signal, density_normalization)

    @normalize_shape(2)
    def emulate_readings_new(self, x, signal, normal_std, k, bins):
        return torch_radon_cuda.emulate_readings_new(x, self.noise_generator, signal, normal_std, k, bins)

    @normalize_shape(2)
    def readings_lookup(self, sensor_readings, lookup_table):
        return torch_radon_cuda.readings_lookup(sensor_readings, lookup_table)

    def set_seed(self, seed=-1):
        if seed < 0:
            seed = np.random.get_state()[1][0]

        self.noise_generator.set_seed(seed)

    def __del__(self):
        self.noise_generator.free()


class Radon(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/parallelbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Parallel Beam geometry.

    :param resolution: The resolution of the input images.
    :param angles: Array containing the list of measuring angles. Can be a Numpy array or a PyTorch tensor.
    :param det_count: Number of rays that will be projected. By default it is = :attr:`resolution`
    :param det_spacing: Distance between two contiguous rays.
    :param clip_to_circle: If True both forward and backward projection will be restricted to pixels inside the circle
        (highlighted in cyan).

    .. note::
        Currently only support resolutions which are multiples of 16.
    """

    def __init__(self, resolution: int, angles, det_count=-1, det_spacing=1.0, clip_to_circle=False):
        if det_count <= 0:
            det_count = resolution

        rays_cfg = RaysCfg(resolution, resolution, det_count, det_spacing, len(angles), clip_to_circle)

        super().__init__(angles, rays_cfg)

        self.det_count = det_count
        self.det_spacing = det_spacing


class RadonFanbeam(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/fanbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Fanbeam geometry.

    :param resolution: The resolution of the input images.
    :param angles: Array containing the list of measuring angles. Can be a Numpy array or a PyTorch tensor.
    :param source_distance: Distance between the source of rays and the center of the image.
    :param det_distance: Distance between the detector plane and the center of the image.
        By default it is =  :attr:`source_distance`.
    :param det_count: Number of rays that will be projected. By default it is = :attr:`resolution`.
    :param det_spacing: Distance between two contiguous rays.
    :param clip_to_circle: If True both forward and backward projection will be restricted to pixels inside the circle
        (highlighted in cyan).

    .. note::
        Currently only support resolutions which are multiples of 16.
    """

    def __init__(self, resolution: int, angles, source_distance: float, det_distance: float = -1, det_count: int = -1,
                 det_spacing: float = -1, clip_to_circle=False):

        if det_count <= 0:
            det_count = resolution

        if det_distance < 0:
            det_distance = source_distance
            det_spacing = 2.0
        if det_spacing < 0:
            det_spacing = (source_distance + det_distance) / source_distance

        rays_cfg = RaysCfg(resolution, resolution, det_count, det_spacing, len(angles), clip_to_circle,
                           source_distance, det_distance)

        super().__init__(angles, rays_cfg)

        self.source_distance = source_distance
        self.det_distance = det_distance
        self.det_count = det_count
        self.det_spacing = det_spacing


def compute_lookup_table(sinogram, signal, normal_std, bins=4096, eps=0.01, eps_prob=0.99, eps_k=0.01, verbose=False):
    s = sinogram.view(-1)
    device = s.device

    eps = np.quantile(sinogram.cpu().numpy(), eps) + eps_k

    # Compute readings normalization value
    if verbose:
        print("Computing readings normalization value")
    k = 0
    for i in range(1, 5000):
        a, b = torch_radon_cuda.compute_ab(s, signal, eps, bins * i)
        if verbose:
            print(a, b)
        if a >= (a + b) * eps_prob:
            k = bins * i
            break
    print("Readings normalization value = ", k // bins)

    # Compute weights for Gaussian error
    scale = k // bins
    weights = []
    for i in range(0, 64):
        t = scipy.stats.norm.cdf((scale - i - 0.5) / normal_std) - scipy.stats.norm.cdf((- i - 0.5) / normal_std)
        if t < 0.005:
            break

        weights.append(t)

    weights = weights[scale:][::-1] + weights
    weights = np.array(weights)

    border_w = np.asarray([scipy.stats.norm.cdf((-x - 0.5) / normal_std) for x in range(scale)])
    border_w = torch.FloatTensor(border_w).to(device)

    log_factorial = np.arange(k + len(weights))
    log_factorial[0] = 1
    log_factorial = np.cumsum(np.log(log_factorial).astype(np.float64)).astype(np.float32)
    log_factorial = torch.Tensor(log_factorial).to(device)

    weights = torch.FloatTensor(weights).to(device)

    lookup, lookup_var = torch_radon_cuda.compute_lookup_table(s, weights, signal, bins, scale, log_factorial, border_w)

    return lookup, lookup_var, scale


class ReadingsLookup:
    def __init__(self, radon, bins=4096, mu=None, sigma=None, ks=None, signals=None, normal_stds=None):
        self.radon = radon
        self.bins = bins

        self.mu = [] if mu is None else mu
        self.sigma = [] if sigma is None else sigma
        self.ks = [] if ks is None else ks

        self.signals = [] if signals is None else signals
        self.normal_stds = [] if normal_stds is None else normal_stds

        self._mu = None
        self._sigma = None
        self._ks = None
        self._signals = None
        self._normal_stds = None
        self._need_repacking = True

    def repack(self, device):
        self._mu = torch.FloatTensor(self.mu).to(device)
        self._sigma = torch.FloatTensor(self.sigma).to(device)
        self._ks = torch.IntTensor(self.ks).to(device)
        self._signals = torch.FloatTensor(self.signals).to(device)
        self._normal_stds = torch.FloatTensor(self.normal_stds).to(device)

    @staticmethod
    def from_file(path, radon):
        obj = np.load(path)

        bins = int(obj["bins"])

        return ReadingsLookup(radon, bins, list(obj["mu"]), list(obj["sigma"]), list(obj["ks"]), list(obj["signals"]),
                              list(obj["normal_stds"]))

    def save(self, path):
        self.repack("cpu")
        np.savez(path, mu=self._mu, sigma=self._sigma, ks=self._ks, signals=self._signals,
                 normal_stds=self._normal_stds, bins=self.bins)

    def add_lookup_table(self, sinogram, signal, normal_std, eps=0.01, eps_prob=0.99, eps_k=0.01, verbose=True):
        lookup, lookup_var, k = compute_lookup_table(sinogram, signal, normal_std, self.bins, eps, eps_prob, eps_k,
                                                     verbose)

        self.mu.append(lookup.cpu().numpy())
        self.sigma.append(lookup_var.cpu().numpy())
        self.ks.append(k)
        self.signals.append(signal)
        self.normal_stds.append(normal_std)
        self._need_repacking = True

    @normalize_shape(2)
    def emulate_readings(self, sinogram, level):
        if self._need_repacking or self._mu.device != sinogram.device:
            self.repack(sinogram.device)

        if isinstance(level, torch.Tensor):
            return torch_radon_cuda.emulate_readings_multilevel(sinogram, self.radon.noise_generator, self._signals,
                                                                self._normal_stds, self._ks, level, self.bins)
        else:
            return torch_radon_cuda.emulate_readings_new(sinogram, self.radon.noise_generator, self.signals[level],
                                                         self.normal_stds[level], self.ks[level], self.bins)

    @normalize_shape(2)
    def lookup(self, readings, level):
        if self._need_repacking or self._mu.device != readings.device:
            self.repack(readings.device)

        if isinstance(level, torch.Tensor):
            mu = torch_radon_cuda.readings_lookup_multilevel(readings, self._mu, level)
            sigma = torch_radon_cuda.readings_lookup_multilevel(readings, self._sigma, level)
        else:
            mu = torch_radon_cuda.readings_lookup(readings, self._mu[level])
            sigma = torch_radon_cuda.readings_lookup(readings, self._sigma[level])

        return mu, sigma

    def random_levels(self, size, device):
        return torch.randint(0, len(self.mu), (size,), device=device, dtype=torch.int32)
