import numpy as np
import torch
from nose.tools import assert_less
from parameterized import parameterized

from torch_radon import Radon
from .astra_wrapper import AstraWrapper
from .utils import generate_random_images, relative_error, circle_mask

device = torch.device('cuda')

full_angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)
limited_angles = np.linspace(0.2 * np.pi, 0.5 * np.pi, 50).astype(np.float32)
sparse_angles = np.linspace(0, 2 * np.pi, 60).astype(np.float32)

params = []  # [(device, 8, 128, full_angles)]
for batch_size in [1, 8, 16]:  # , 64, 128]:  # , 256, 512]:
    for image_size in [128, 128 + 32, 256]:  # , 512]:
        for angles in [full_angles, limited_angles, sparse_angles]:
            for spacing in [1.0, 0.5, 1.3, 2.0]:
                for clip_to_circle in [False, True]:
                    params.append((device, batch_size, image_size, angles, spacing, clip_to_circle))

half_params = [x for x in params if x[1] % 4 == 0]


@parameterized(params)
def test_error(device, batch_size, image_size, angles, spacing, clip_to_circle):
    # generate random images
    x = generate_random_images(batch_size, image_size, masked=clip_to_circle)

    # astra
    astra = AstraWrapper(angles)

    astra_fp_id, astra_fp = astra.forward(x, spacing)
    astra_bp = astra.backproject(astra_fp_id, image_size, batch_size)
    if clip_to_circle:
        astra_bp *= circle_mask(image_size)

    # our implementation
    radon = Radon(image_size, angles, det_spacing=spacing, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device)

    our_fp = radon.forward(x)
    our_bp = radon.backprojection(our_fp)

    forward_error = relative_error(astra_fp, our_fp.cpu().numpy())
    back_error = relative_error(astra_bp, our_bp.cpu().numpy())

    # if forward_error > 10:
    #     plt.imshow(astra_fp[0])
    #     plt.figure()
    #     plt.imshow(our_fp[0].cpu().numpy())
    #     plt.show()

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")
    # TODO better checks
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)


@parameterized(half_params)
def test_half(device, batch_size, image_size, angles, spacing, clip_to_circle):
    # generate random images
    x = generate_random_images(batch_size, image_size, masked=clip_to_circle)

    # our implementation
    radon = Radon(image_size, angles, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device)

    sinogram = radon.forward(x)
    single_precision = radon.backprojection(sinogram)

    h_sino = radon.forward(x.half())
    half_precision = radon.backprojection(h_sino)

    forward_error = relative_error(sinogram.cpu().numpy(), h_sino.cpu().numpy())
    back_error = relative_error(single_precision.cpu().numpy(), half_precision.cpu().numpy())

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")

    assert_less(forward_error, 1e-3)
    assert_less(back_error, 1e-3)
