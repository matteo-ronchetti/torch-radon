import numpy as np
import torch


def normalize_shape(f):
    def inner(self, x, *args, **kwargs):
        # if input has shape BATCH x CHANNELS x W x H reshape to BATCH*CHANNELS x W x H
        old_shape = None
        if len(x.size()) == 4:
            old_shape = x.size()
            x = x.view(-1, old_shape[-2], old_shape[-1])

        y = f(self, x, *args, **kwargs)

        # return to old shape
        if old_shape is not None:
            y = y.view(old_shape[0], old_shape[1], -1, old_shape[-1])

        return y

    return inner


def compute_rays(resolution):
    s = resolution // 2
    locations = np.arange(2 * s) - s + 0.5
    ys = np.sqrt(s ** 2 - locations ** 2) - 0.5
    locations = locations.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    rays = np.hstack((locations, -ys, locations, ys))
    return torch.FloatTensor(rays)
