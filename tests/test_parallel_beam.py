import numpy as np
import torch
from nose.tools import assert_less
from parameterized import parameterized

from torch_radon import Radon
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
                    for clip_to_circle in [False, True]:
                        params.append((device, batch_size, image_size, angles, spacing, det_count, clip_to_circle))

half_params = [x for x in params if x[1] % 4 == 0]


@parameterized(params)
def test_error(device, batch_size, image_size, angles, spacing, det_count, clip_to_circle):
    # generate random images
    det_count = int(det_count * image_size)
    mask_radius = det_count / 2.0 if clip_to_circle else -1
    x = generate_random_images(batch_size, image_size, mask_radius)

    # astra
    astra = AstraWrapper(angles)

    astra_fp_id, astra_fp = astra.forward(x, spacing, det_count)
    astra_bp = astra.backproject(astra_fp_id, image_size, batch_size)
    if clip_to_circle:
        astra_bp *= circle_mask(image_size, mask_radius)

    # our implementation
    radon = Radon(image_size, angles, det_spacing=spacing, det_count=det_count, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device)

    our_fp = radon.forward(x)
    our_bp = radon.backprojection(our_fp)

    forward_error = relative_error(astra_fp, our_fp.cpu().numpy())
    back_error = relative_error(astra_bp, our_bp.cpu().numpy())

    if back_error > 1e-2:
        plt.figure()
        plt.title('ASTRA backprojection')
        plt.imshow(astra_bp[0])
        plt.colorbar()
        plt.figure()
        plt.title('torch-radon difference')
        plt.imshow(np.abs(our_bp[0].cpu().numpy() - astra_bp[0])/astra_bp[0])
        plt.colorbar()
        plt.show()

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, det_count: {det_count}, circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")
    # TODO better checks
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)


@parameterized(half_params)
def test_half(device, batch_size, image_size, angles, spacing, det_count, clip_to_circle):
    # generate random images
    det_count = int(det_count * image_size)
    mask_radius = det_count / 2.0 if clip_to_circle else -1
    x = generate_random_images(batch_size, image_size, mask_radius)

    # scale used to avoid overflow in BP
    bp_scale = np.pi / len(angles)

    # our implementation
    radon = Radon(image_size, angles, det_spacing=spacing, det_count=det_count, clip_to_circle=clip_to_circle)
    x = torch.FloatTensor(x).to(device)

    sinogram = radon.forward(x)
    single_precision = radon.backprojection(sinogram)

    h_sino = radon.forward(x.half())
    half_precision = radon.backprojection(h_sino * bp_scale)

    forward_error = relative_error(sinogram.cpu().numpy(), h_sino.cpu().numpy())
    back_error = relative_error(single_precision.cpu().numpy(), half_precision.cpu().float().numpy() / bp_scale)

    print(
        f"batch: {batch_size}, size: {image_size}, angles: {len(angles)}, spacing: {spacing}, circle: {clip_to_circle}, forward: {forward_error}, back: {back_error}")

    assert_less(forward_error, 1e-3)
    assert_less(back_error, 1e-3)


def test_simple_integrals(image_size=17):
    """Check that the forward radon operator works correctly at 0 and PI/2.

    When we project at angles 0 and PI/2, the foward operator should be the
    same as taking the sum over the object array along each axis.
    """
    angles = torch.tensor(
        [0.0, -np.pi / 2, np.pi, np.pi / 2],
        dtype=torch.float32,
        device='cuda',
    )
    radon = Radon(
        resolution=image_size,
        angles=angles,
        det_spacing=1.0,
        det_count=image_size,
        clip_to_circle=False,
    )

    original = torch.zeros(
        image_size,
        image_size,
        dtype=torch.float32,
        device='cuda',
    )
    original[image_size // 4, :] += 1
    # original[:, image_size // 2] += 1

    data = radon.forward(original)
    data0 = torch.sum(original, axis=0)
    data1 = torch.sum(original, axis=1)

    print('\n', data[0].cpu().numpy())
    print(data0.cpu().numpy())
    print('\n', data[1].cpu().numpy())
    print(data1.cpu().numpy())
    print('\n', data[2].cpu().numpy())
    print(data0.cpu().numpy()[::-1])
    print('\n', data[3].cpu().numpy())
    print(data1.cpu().numpy()[::-1])

    # torch.testing.assert_allclose(data[0], data0)
    # torch.testing.assert_allclose(data[1], data1)
    # torch.testing.assert_allclose(data[2], data0)
    # torch.testing.assert_allclose(data[3], data1)


def test_simple_back(image_size=17):

    data = torch.zeros(4, image_size, device='cuda')
    data[:, image_size // 4] = torch.tensor([1, 2, 3, 4], device='cuda')

    angles = torch.tensor(
        [0.0, np.pi / 2, np.pi, -np.pi / 2],
        dtype=torch.float32,
        device='cuda',
    )
    radon = Radon(
        resolution=image_size,
        angles=angles,
        det_spacing=1.0,
        det_count=image_size,
        clip_to_circle=False,
    )

    original = radon.backward(data)
    print()
    print(original)
