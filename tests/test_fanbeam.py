from .utils import generate_random_images, relative_error, circle_mask
import astra
from nose.tools import assert_less, assert_equal
import torch
import numpy as np
from torch_radon import RadonFanbeam
from parameterized import parameterized
import matplotlib.pyplot as plt

device = torch.device('cuda')

full_angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)
limited_angles = np.linspace(0.2 * np.pi, 0.5 * np.pi, 50).astype(np.float32)
sparse_angles = np.linspace(0, 2 * np.pi, 60).astype(np.float32)

params = []
for batch_size in [1, 8]:
    for image_size in [128, 128 + 32]:
        for angles in [full_angles, limited_angles, sparse_angles]:
            for spacing in [1.0, 0.5, 1.3, 2.0]:
                for distances in [(1.2, 1.2), (2.0, 2.0), (1.2, 3.0)]:
                    for det_count in [1.0, 1.5]:
                        for clip_to_circle in [False, True]:
                            params.append((device, batch_size, image_size, angles, spacing, distances, det_count, clip_to_circle))

half_params = [x for x in params if x[1] % 4 == 0]


@parameterized(params)
def test_fanbeam_error(device, batch_size, image_size, angles, spacing, distances, det_count, clip_to_circle):
    # generate random images
    # generate random images
    det_count = int(det_count * image_size)
    mask_radius = det_count / 2.0 if clip_to_circle else -1
    x = generate_random_images(1, image_size, mask_radius)[0]

    s_dist, d_dist = distances
    s_dist *= image_size
    d_dist *= image_size

    # astra
    vol_geom = astra.create_vol_geom(x.shape[0], x.shape[1])
    proj_geom = astra.create_proj_geom('fanflat', spacing, det_count, angles, s_dist, d_dist)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    id, astra_y = astra.create_sino(x, proj_id)
    _, astra_bp = astra.create_backprojection(astra_y, proj_id)
    if clip_to_circle:
        astra_bp *= circle_mask(image_size, mask_radius)

    # TODO clean astra structures

    # our implementation
    radon = RadonFanbeam(image_size, angles, s_dist, d_dist, det_count=det_count, det_spacing=spacing, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device).view(1, x.shape[0], x.shape[1])
    # repeat data to fill batch size
    x = torch.cat([x] * batch_size, dim=0)

    our_fp = radon.forward(x)
    our_bp = radon.backprojection(our_fp)

    forward_error = relative_error(astra_y, our_fp[0].cpu().numpy())
    back_error = relative_error(astra_bp, our_bp[0].cpu().numpy())

    # if back_error > 5e-3:
    #     plt.imshow(astra_bp)
    #     plt.figure()
    #     plt.imshow(our_bp[0].cpu().numpy())
    #     plt.show()

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, distances: {distances} circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")
    # TODO better checks
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)


@parameterized(half_params)
def test_half(device, batch_size, image_size, angles, spacing, distances, det_count, clip_to_circle):
    # generate random images
    det_count = int(det_count * image_size)
    mask_radius = det_count / 2.0 if clip_to_circle else -1
    x = generate_random_images(batch_size, image_size, mask_radius)

    s_dist, d_dist = distances
    s_dist *= image_size
    d_dist *= image_size

    # our implementation
    radon = RadonFanbeam(image_size, angles, s_dist, d_dist, det_count=det_count, det_spacing=spacing, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device)

    # divide by len(angles) to avoid half-precision overflow
    sinogram = radon.forward(x) / len(angles)
    single_precision = radon.backprojection(sinogram)

    h_sino = radon.forward(x.half()) / len(angles)
    half_precision = radon.backprojection(h_sino)
    print(torch.min(half_precision).item(), torch.max(half_precision).item())

    forward_error = relative_error(sinogram.cpu().numpy(), h_sino.cpu().numpy())
    back_error = relative_error(single_precision.cpu().numpy(), half_precision.cpu().numpy())

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")

    assert_less(forward_error, 1e-3)
    assert_less(back_error, 1e-3)
