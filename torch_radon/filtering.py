import numpy as np
import torch
from functools import lru_cache

try:
    import scipy.fft

    fftmodule = scipy.fft
except ImportError:
    import numpy.fft

    fftmodule = numpy.fft


class FourierFilters:
    def __init__(self):
        pass

    def get(self, size: int, filter_name: str, device):
        filter_name = filter_name.lower()
        ff = self.construct_fourier_filter(size, filter_name).view(1, 1, -1).to(device)
        return ff

    @staticmethod
    @lru_cache(maxsize=128)
    def construct_fourier_filter(size, filter_name):
        """Construct the Fourier filter.

        This computation lessens artifacts and removes a small bias as
        explained in [1], Chap 3. Equation 61.

        Parameters
        ----------
        size: int
            filter size. Must be even.
        filter_name: str
            Filter used in frequency domain filtering. Filters available:
            ram-lak (ramp), shepp-logan, cosine, hamming, hann.

        Returns
        -------
        fourier_filter: ndarray
            The computed Fourier filter.

        References
        ----------
        .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
               Imaging", IEEE Press 1988.

        """

        # Initial computations
        n = np.concatenate((np.arange(1, size // 2 + 1, 2, dtype=np.int),
                            np.arange(size // 2 - 1, 0, -2, dtype=np.int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2
        fourier_filter = 2 * np.real(fftmodule.fft(f))  # ramp filter

        # Frequency domain filter adjustments
        filter_functions = {
            "ramp": lambda x: x,
            "ram-lak": lambda x: x,
            "shepp-logan": lambda x: x[1:] * np.sin(np.pi * fftmodule.fftfreq(size)[1:]) / np.pi * fftmodule.fftfreq(size)[1:],
            "cosine": lambda x: x * fftmodule.fftshift(np.sin(np.linspace(0, np.pi, size, endpoint=False))),
            "hamming": lambda x: x * fftmodule.fftshift(np.hamming(size)),
            "hann": lambda x: x * fftmodule.fftshift(np.hanning(size))
        }
        if filter_name not in filter_functions:
            raise ValueError(f"[TorchRadon] Error, unknown filter type '{filter_name}', available filters are: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'")
        
        fourier_filter = filter_functions[filter_name](fourier_filter)

        # Return the first n//2 + 1 elements to match the output of rfft
        return torch.FloatTensor(fourier_filter[:size//2 + 1])
