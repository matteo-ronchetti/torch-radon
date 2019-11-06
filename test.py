import torch
import cv2
import numpy as np


def rot(a):
    return np.asarray([
        [np.cos(a), -np.sin(a)],
        [np.sin(a), np.cos(a)]
    ], dtype=np.float32)


# x = np.abs(np.random.randn(128, 128)).astype(np.float32)

s = 64

lines = np.asarray([
    [s, 0, 0, 1],
    [0, s, 1, 0],
    [-s, 0, 0, 1],
    [0, -s, 1, 0],
], dtype=np.float32)

angles = np.linspace(0, 2*np.pi, 10)
locations = np.arange(2*s) - s + 0.5

ys = np.sqrt((s-0.5)**2 - locations**2)
locations = locations.reshape(-1, 1)
ys = ys.reshape(-1, 1)
rays = np.hstack((locations, -ys, locations, ys))


# for i, angle in enumerate(angles):
#     R = rot(angle)
#     x = (R @ lines.reshape(-1, 2).T).T.reshape(-1, 4)
#     for t in range(2*s):
#         tt = t - s
#         y = x[:, 1] - (x[:, 0] - tt) / x[:, 2] * x[:, 3]
#         _, sy, ey, _ = np.sort(y)
