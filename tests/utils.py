import numpy as np
import cv2


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / np.linalg.norm(ref)

def circle_mask(size):
    radius = size/2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - radius + 0.5) ** 2 + (c1 - radius + 0.5) ** 2) <= radius ** 2

def generate_random_images(n, size):
    # create a circular mask
   # cv2.imwrite("mask.png", mask.astype(np.uint8)*255)
    mask = circle_mask(size)
    
    # generate images and apply mask
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)
    batch *= mask
        
    return batch
