from .AlphaTransform import AlphaShearletTransform
from .fourier_util import my_ifft_shift
from ..utils import normalize_shape
import numpy as np
import torch
import os


class Shearlet:
    def __init__(self, width, height, alphas, cache=None):
        cache_name = f"{width}_{height}_{alphas}.npy"
        if cache is not None:
            if not os.path.exists(cache):
                os.makedirs(cache)

            cache_file = os.path.join(cache, cache_name)
            if os.path.exists(cache_file):
                self.shifted_spectrograms = np.load(cache_file)
            else:
                alpha_shearlet = AlphaShearletTransform(width, height, alphas, real=True, parseval=True)
                self.shifted_spectrograms = np.asarray([my_ifft_shift(spec) for spec in alpha_shearlet.spectrograms])
                np.save(cache_file, self.shifted_spectrograms)
        else:
            alpha_shearlet = AlphaShearletTransform(width, height, alphas, real=True, parseval=True)
            scales = [0] + [x[0] for x in alpha_shearlet.indices[1:]]
            self.scales = np.asarray(scales)
            self.shifted_spectrograms = np.asarray([my_ifft_shift(spec) for spec in alpha_shearlet.spectrograms])

        self.scales = torch.FloatTensor(self.scales)
        self.shifted_spectrograms = torch.FloatTensor(self.shifted_spectrograms)

    def _move_parameters_to_device(self, device):
        if device != self.shifted_spectrograms.device:
            self.shifted_spectrograms = self.shifted_spectrograms.to(device)

    # @normalize_shape
    def forward(self, x):
        self._move_parameters_to_device(x.device)

        c = torch.rfft(x, 2, normalized=True, onesided=False)

        cs = torch.einsum("fij,bijc->bfijc", self.shifted_spectrograms, c)
        return torch.irfft(cs, 2, normalized=True, onesided=False)

    # @normalize_shape
    def backward(self, cs):
        cs_fft = torch.rfft(cs, 2, normalized=True, onesided=False)
        # print(cs.size(), cs_fft.size())
        res = torch.einsum("fij,bfijc->bijc", self.shifted_spectrograms, cs_fft)
        return torch.irfft(res, 2, normalized=True, onesided=False)

# device = torch.device("cuda")
# im = np.random.uniform(0, 1, (512, 512))
# # alpha_shearlet = AlphaShearletTransform(im.shape[1], im.shape[0], [0.5] * 5, real=True, parseval=True)
# shearlet = Shearlet(im.shape[1], im.shape[0], [0.5] * 5, cache="cache")
#
# # coeff_ = alpha_shearlet.transform(im, do_norm=False)
# # rec_ = alpha_shearlet.adjoint_transform(coeff_, do_norm=False)
#
# with torch.no_grad():
#     x = torch.DoubleTensor(im).to(device)
#     coeff = shearlet.forward(x)
#     rec = shearlet.backward(coeff)
#     print(torch.norm(rec - x)/torch.norm(x))
#
# # print(np.allclose(coeff_, coeff.cpu().numpy()))
# # print(np.allclose(rec_, rec.cpu().numpy()))
# #
# # print("Alpha forward", timeit.timeit(lambda: alpha_shearlet.transform(im, do_norm=False), number=10))
# # print("Alpha backward", timeit.timeit(lambda: alpha_shearlet.adjoint_transform(coeff_, do_norm=False), number=10))
# # print("My forward", timeit.timeit(lambda: shearlet.forward(x), number=30))
# # print("My backward", timeit.timeit(lambda: shearlet.backward(coeff), number=30))
