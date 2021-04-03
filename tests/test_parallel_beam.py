import numpy as np
import torch
from nose.tools import assert_less
from parameterized import parameterized

from torch_radon import Radon, Projection
from .astra_wrapper import AstraWrapper
from .utils import generate_random_images, relative_error, circle_mask
import matplotlib.pyplot as plt

device = torch.device('cuda')

full_angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)
limited_angles = np.linspace(0.2 * np.pi, 0.5 * np.pi, 50).astype(np.float32)
sparse_angles = np.linspace(0, 2 * np.pi, 60).astype(np.float32)
many_angles = np.linspace(0, 2 * np.pi, 800).astype(np.float32)

params = []  # [(device, 8, 128, full_angles)]
for batch_size in [1, 8, 16]:  # , 64, 128]:  # , 256, 512]:
    for image_size in [128, 245, 256]:  # , 512]:
        for angles in [full_angles, limited_angles, sparse_angles, many_angles]:
            for spacing in [1.0, 0.5, 1.3, 2.0]:
                for det_count in [1.0, 1.5]:
                    params.append((device, batch_size, image_size, angles, spacing, det_count))

half_params = [x for x in params if x[1] % 4 == 0]


@parameterized(params)
def test_error(device, batch_size, image_size, angles, spacing, det_count):
    # generate random images
    det_count = int(det_count * image_size)
    x = generate_random_images(batch_size, image_size)

    # astra
    astra = AstraWrapper(angles)

    astra_fp_id, astra_fp = astra.forward(x, spacing, det_count)
    astra_bp = astra.backproject(astra_fp_id, image_size, batch_size)

    # our implementation
    projection = Projection.parallel_beam(det_count=det_count, det_spacing=spacing)
    radon = Radon(angles, image_size, projection)
    x = torch.FloatTensor(x).to(device)

    our_fp = radon.forward(x)
    our_bp = radon.backprojection(our_fp)

    forward_error = relative_error(astra_fp, our_fp.cpu().numpy())
    back_error = relative_error(astra_bp, our_bp.cpu().numpy())

    if forward_error > 1e-2:
        fig, ax = plt.subplots(2, 3)
        ax = ax.ravel()
        ax[0].imshow(astra_fp[0])
        ax[1].imshow(our_fp[0].cpu().numpy())
        ax[2].imshow(astra_fp[0] - our_fp[0].cpu().numpy())
        ax[3].imshow(astra_bp[0])
        ax[4].imshow(our_bp[0].cpu().numpy())
        ax[5].imshow(astra_bp[0] - our_bp[0].cpu().numpy())
        plt.show()

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, det_count: {det_count}, forward: {forward_error}, back: {back_error}")
    # TODO better checks
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)


@parameterized(half_params)
def test_half(device, batch_size, image_size, angles, spacing, det_count):
    # generate random images
    det_count = int(det_count * image_size)
    x = generate_random_images(batch_size, image_size)

    # scale used to avoid overflow in BP
    bp_scale = np.pi / len(angles)

    # our implementation
    projection = Projection.parallel_beam(det_count=det_count, det_spacing=spacing)
    radon = Radon(angles, image_size, projection)
    x = torch.FloatTensor(x).to(device)

    sinogram = radon.forward(x)
    single_precision = radon.backprojection(sinogram)

    h_sino = radon.forward(x.half())
    half_precision = radon.backprojection(h_sino * bp_scale)

    forward_error = relative_error(sinogram.cpu().numpy(), h_sino.cpu().numpy())
    back_error = relative_error(single_precision.cpu().numpy(), half_precision.cpu().float().numpy() / bp_scale)

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, forward: {forward_error}, back: {back_error}")

    assert_less(forward_error, 1e-3)
    assert_less(back_error, 1e-3)
