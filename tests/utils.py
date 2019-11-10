import numpy as np


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / np.linalg.norm(ref)


def generate_random_images(n, size):
    # create a circular mask
    c0, c1 = np.ogrid[0:size, 0:size]
    mask = ((c0 - size // 2) ** 2 + (c1 - size // 2) ** 2) <= (size//2 - 2) ** 2

    # generate images and apply mask
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)
    batch *= mask

    return batch
