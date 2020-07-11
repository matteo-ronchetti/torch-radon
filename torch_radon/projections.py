import numpy as np
import torch


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
