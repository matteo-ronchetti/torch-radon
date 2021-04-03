import numpy as np
import torch
from nose.tools import assert_equal

from torch_radon import Radon


# def test_noise():
#     device = torch.device('cuda')
#
#     x = torch.FloatTensor(3, 5, 64, 64).to(device)
#     lookup_table = torch.FloatTensor(128, 64).to(device)
#     x.requires_grad = True
#     angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, 10).astype(np.float32))
#
#     radon = Radon(angles, 64)
#
#     sinogram = radon.forward(x)
#     assert_equal(sinogram.size(), (3, 5, 10, 64))
#
#     readings = radon.emulate_readings(sinogram, 5, 10.0)
#     assert_equal(readings.size(), (3, 5, 10, 64))
#     assert_equal(readings.dtype, torch.int32)
#
#     y = radon.readings_lookup(readings, lookup_table)
#     assert_equal(y.size(), (3, 5, 10, 64))
#     assert_equal(y.dtype, torch.float32)
