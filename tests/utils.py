import numpy as np


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / (np.linalg.norm(ref) + 1e-6)


def circle_mask(size, radius):
    center = (size - 1) / 2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - center) ** 2 + (c1 - center) ** 2) <= radius ** 2


def generate_random_images(n, size, mask_radius=-1):
    # generate images
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)

    if mask_radius > 0:
        # create and apply circular mask
        mask = circle_mask(size, mask_radius)
        batch *= mask

    return batch
