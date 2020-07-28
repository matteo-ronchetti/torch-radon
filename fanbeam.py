from torch_radon import Radon, RadonFanbeam
from tests.astra_wrapper import AstraWrapper
import astra
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Projection:
    def __init__(self, resolution, rays, clip_to_circle=True):
        assert resolution % 16 == 0, "Resolution must be multiple of 16"
        self.resolution = resolution

        if clip_to_circle:
            rays = self.clip_to_circle(resolution, rays)
        self.rays = torch.FloatTensor(rays)

    @staticmethod
    def clip_to_circle(resolution, rays):
        s = resolution // 2

        # intersect rays with circle
        a = (rays[:, 2] - rays[:, 0]) ** 2 + (rays[:, 3] - rays[:, 1]) ** 2
        b = 2 * (rays[:, 0] * (rays[:, 2] - rays[:, 0]) + rays[:, 1] * (rays[:, 3] - rays[:, 1]))
        c = rays[:, 0] ** 2 + rays[:, 1] ** 2 - s ** 2

        # min_clip to 1 to avoid getting empty rays
        delta_sqrt = np.sqrt(np.clip(b ** 2 - 4 * a * c, 1, 1e30))
        x1 = (-b + delta_sqrt) / (2 * a)
        x2 = (-b - delta_sqrt) / (2 * a)

        old_rays = rays.copy()
        rays[:, 0] = old_rays[:, 0] + x1 * (old_rays[:, 2] - old_rays[:, 0])
        rays[:, 1] = old_rays[:, 1] + x1 * (old_rays[:, 3] - old_rays[:, 1])
        rays[:, 2] = old_rays[:, 0] + x2 * (old_rays[:, 2] - old_rays[:, 0])
        rays[:, 3] = old_rays[:, 1] + x2 * (old_rays[:, 3] - old_rays[:, 1])

        return rays


class FanBeamProjection(Projection):
    def __init__(self, resolution, source_distance: float, det_distance: float = -1, det_width: float = -1,
                 clip_to_circle=True):
        if det_distance < 0:
            det_distance = source_distance
            det_width = 2.0
        if det_width < 0:
            det_width = (source_distance + det_distance) / source_distance

        s = resolution // 2
        locations = np.arange(2 * s) - s + 0.5

        locations = locations.reshape(-1, 1)

        zeros = np.zeros((resolution, 1))
        ones = np.ones((resolution, 1))
        rays = np.hstack((zeros, source_distance * ones, det_width * locations, -det_distance * ones))

        super().__init__(resolution, rays, clip_to_circle)


class ParallelBeamProjection(Projection):
    def __init__(self, resolution):
        s = resolution // 2
        locations = (np.arange(2 * s) - s + 0.5).reshape(-1, 1)
        ys = np.ones((resolution, 1)) * resolution
        rays = np.hstack((locations, ys, locations, -ys))

        super().__init__(resolution, rays)


def circle_mask(size):
    radius = (size - 1) / 2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - radius) ** 2 + (c1 - radius) ** 2) <= (radius) ** 2


def generate_random_images(n, size):
    # create a circular mask
    # cv2.imwrite("mask.png", mask.astype(np.uint8)*255)
    mask = circle_mask(size)

    # generate images and apply mask
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)
    #batch[0] = cv2.GaussianBlur(batch[0], (3, 3), -1)
    #batch *= mask

    return batch


res = 160
angles = np.linspace(0, np.pi, 60).astype(np.float32)  # np.linspace(0, 2 * np.pi, 180).astype(np.float32)

#x = np.load("examples/phantom.npy")
x = generate_random_images(16, res)[0]
# plt.imshow(x)
# plt.show()
print(x.shape)


source_distance = 1.2*res
det_distance = 3.0*res
det_width = 2.0

# astra
# astra = AstraWrapper(angles)
#
# astra_fp_id, astra_y = astra.forward(x, det_width)
# astra_bp = astra.backproject(astra_fp_id, res, 1)

vol_geom = astra.create_vol_geom(x.shape[0], x.shape[1])
proj_geom = astra.create_proj_geom('fanflat', det_width, x.shape[0], angles, source_distance, det_distance)
#proj_geom = astra.create_proj_geom('parallel', det_width, x.shape[0], angles)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

id, astra_y = astra.create_sino(x, proj_id)
_, astra_bp = astra.create_backprojection(astra_y, proj_id)

plt.imshow(astra_bp)
# #
# # plt.figure()
# # # projection = FanBeamProjection(x.shape[0], source_distance, det_distance, det_width)
# # # projection = ParallelBeamProjection(x.shape[0])
radon = RadonFanbeam(x.shape[-1], angles, source_distance, det_distance, det_spacing=det_width)
print(source_distance + det_distance)
# radon.rays = fan_beam_rays(x.shape[0], source_distance, det_distance, det_width, clip_to_circle=True)
# print(radon.rays.size())

x = torch.FloatTensor(x).cuda().view(1, x.shape[0], x.shape[1])
# repeat data to fill batch size
x = torch.cat([x]*8, dim=0)
y = radon.forward(x)
bp = radon.backprojection(y)

print(np.linalg.norm(astra_y - y.cpu().numpy()) / np.linalg.norm(astra_y))
scale = np.mean(astra_bp) / np.mean(bp[0].cpu().numpy())
print(scale)
print(np.linalg.norm(astra_bp - bp[0].cpu().numpy()) / np.linalg.norm(astra_bp))

plt.figure()
plt.imshow(bp[0].cpu().numpy())
plt.show()
