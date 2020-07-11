from torch_radon import Radon
import astra
import torch
import numpy as np
import matplotlib.pyplot as plt


def fan_beam_rays(resolution, source_distance, det_distance=-1, det_width=-1, clip_to_circle=True):
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

    if clip_to_circle:
        # intersect rays with circle
        a = (rays[:, 2] - rays[:, 0]) ** 2 + (rays[:, 3] - rays[:, 1]) ** 2
        b = 2 * (rays[:, 0] * (rays[:, 2] - rays[:, 0]) + rays[:, 1] * (rays[:, 3] - rays[:, 1]))
        c = rays[:, 0] ** 2 + rays[:, 1] ** 2 - s ** 2

        delta_sqrt = np.sqrt(b ** 2 - 4 * a * c)
        x1 = (-b + delta_sqrt) / (2 * a)
        x2 = (-b - delta_sqrt) / (2 * a)

        old_rays = rays.copy()
        rays[:, 0] = old_rays[:, 0] + x1 * (old_rays[:, 2] - old_rays[:, 0])
        rays[:, 1] = old_rays[:, 1] + x1 * (old_rays[:, 3] - old_rays[:, 1])
        rays[:, 2] = old_rays[:, 0] + x2 * (old_rays[:, 2] - old_rays[:, 0])
        rays[:, 3] = old_rays[:, 1] + x2 * (old_rays[:, 3] - old_rays[:, 1])

    return torch.FloatTensor(rays)


angles = np.linspace(0, 2 * np.pi, 512).astype(np.float32)

x = np.load("examples/phantom.npy")

source_distance = 512
det_distance = 1024
det_width = 1.5

vol_geom = astra.create_vol_geom(x.shape[0], x.shape[1])
proj_geom = astra.create_proj_geom('fanflat', det_width, x.shape[0], -angles, source_distance, det_distance)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

id, astra_y = astra.create_sino(x, proj_id)

plt.imshow(astra_y)

plt.figure()
radon = Radon(x.shape[0], angles)

radon.rays = fan_beam_rays(x.shape[0], source_distance, det_distance, det_width, clip_to_circle=True)
print(radon.rays.size())

y = radon.forward(torch.FloatTensor(x).cuda().view(1, 512, 512))

print(np.linalg.norm(astra_y - y[0].cpu().numpy()) / np.linalg.norm(astra_y))

plt.imshow(y[0].cpu())
plt.show()
