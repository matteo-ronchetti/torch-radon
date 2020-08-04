import numpy as np


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / (np.linalg.norm(ref) + 1e-6)


def circle_mask(size):
    radius = (size - 1) / 2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - radius) ** 2 + (c1 - radius) ** 2) <= (size / 2.0) ** 2


def generate_random_images(n, size, masked=False):
    # generate images
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)

    if masked:
        # create and apply circular mask
        mask = circle_mask(size)
        batch *= mask

    return batch
