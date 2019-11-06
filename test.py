import torch
import cv2
import numpy as np
import radon


def rot(a):
    return np.asarray([
        [np.cos(a), -np.sin(a)],
        [np.sin(a), np.cos(a)]
    ], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        return torch.device("cpu")


device = get_device()

img = cv2.imread("phantom.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

x = torch.FloatTensor(img).to(device)
y = radon.forward(x).cpu().numpy()

print(np.min(y), np.max(y))
y -= np.min(y)
y /= np.max(y) / 255
y = y.astype(np.uint8)

cv2.imwrite("res.png", y)

# s = 64
#
# lines = np.asarray([
#     [s, 0, 0, 1],
#     [0, s, 1, 0],
#     [-s, 0, 0, 1],
#     [0, -s, 1, 0],
# ], dtype=np.float32)
#
# angles = np.linspace(0, 2*np.pi, 10)
# locations = np.arange(2*s) - s + 0.5
#
# ys = np.sqrt((s-0.5)**2 - locations**2)
# locations = locations.reshape(-1, 1)
# ys = ys.reshape(-1, 1)
# rays = np.hstack((locations, -ys, locations, ys))
