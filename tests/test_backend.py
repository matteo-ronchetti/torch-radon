from .utils import generate_random_images, relative_error, circle_mask
from .astra_wrapper import AstraWrapper
from nose.tools import assert_less, assert_equal
import torch
import numpy as np
from torch_radon import Radon
from parameterized import parameterized

device = torch.device('cuda')

full_angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)
limited_angles = np.linspace(0.2 * np.pi, 0.5 * np.pi, 40).astype(np.float32)
sparse_angles = np.linspace(0, 2 * np.pi, 20).astype(np.float32)

params = []  # [(device, 8, 128, full_angles)]
for batch_size in [1, 8, 16, 32]:  # , 64, 128]:  # , 256, 512]:
    for image_size in [32, 128, 256]:  # , 512]:
        for angles in [full_angles, limited_angles, sparse_angles]:
            params.append((device, batch_size, image_size, angles))

half_params = []  # [(device, 8, 128, full_angles)]
for batch_size in [8, 16]:  # , 64, 128]:  # , 256, 512]:
    for image_size in [128, 256]:  # , 512]:
        for angles in [full_angles, limited_angles, sparse_angles]:
            half_params.append((device, batch_size, image_size, angles))


@parameterized(params)
def test_error(device, batch_size, image_size, angles):
    # generate random images
    x = generate_random_images(batch_size, image_size)

    # astra
    astra = AstraWrapper(angles)

    astra_fp_id, astra_fp = astra.forward(x)
    astra_bp = astra.backproject(astra_fp_id, image_size, batch_size)
    # astra_bp *= circle_mask(image_size)

    # our implementation
    radon = Radon(image_size, angles)
    x = torch.FloatTensor(x).to(device)

    our_fp = radon.forward(x)
    our_bp = radon.backprojection(our_fp, extend=True)

    forward_error = relative_error(astra_fp, our_fp.cpu().numpy())
    back_error = relative_error(astra_bp, our_bp.cpu().numpy())

    print(batch_size, image_size, len(angles), forward_error, back_error)
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)


@parameterized(half_params)
def test_half(device, batch_size, image_size, angles):
    # generate random images
    x = generate_random_images(batch_size, image_size)

    # our implementation
    radon = Radon(image_size, angles).to(device)
    x = torch.FloatTensor(x).to(device)

    sinogram = radon.forward(x)
    single_precision = radon.backprojection(sinogram, extend=True)

    h_sino = radon.forward(x.half())
    half_precision = radon.backprojection(h_sino, extend=True)

    forward_error = relative_error(sinogram.cpu().numpy(), h_sino.cpu().numpy())
    back_error = relative_error(single_precision.cpu().numpy(), half_precision.cpu().numpy())

    print(batch_size, image_size, len(angles), forward_error, back_error)

    assert_less(forward_error, 1e-3)
    assert_less(back_error, 1e-3)


def test_noise():
    device = torch.device('cuda')

    x = torch.FloatTensor(3, 5, 64, 64).to(device)
    lookup_table = torch.FloatTensor(128, 64).to(device)
    x.requires_grad = True
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, 10).astype(np.float32))

    radon = Radon(64, angles).to(device)

    sinogram = radon.forward(x)
    assert_equal(sinogram.size(), (3, 5, 10, 64))

    readings = radon.emulate_readings(sinogram, 5, 10.0)
    assert_equal(readings.size(), (3, 5, 10, 64))
    assert_equal(readings.dtype, torch.int32)

    y = radon.readings_lookup(readings, lookup_table)
    assert_equal(y.size(), (3, 5, 10, 64))
    assert_equal(y.dtype, torch.float32)
