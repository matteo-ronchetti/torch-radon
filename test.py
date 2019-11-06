import torch
import cv2
import numpy as np
import radon
import time
import astra

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

# read test image
img = cv2.imread("phantom.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

angles = np.linspace(0, 2*np.pi, 180).astype(np.float32)

# compute rays
s = img.shape[0] // 2
locations = np.arange(2*s) - s + 0.5
print(locations[0], locations[-1])
ys = np.sqrt(s**2 - locations**2)
locations = locations.reshape(-1, 1)
ys = ys.reshape(-1, 1)
rays = np.hstack((locations, -ys, locations, ys))
print("Rays shape", rays.shape)

# move to gpu
x = torch.FloatTensor(img).to(device)
rays = torch.FloatTensor(rays).to(device)
angles = torch.FloatTensor(angles).to(device)

s = time.time()
y = radon.forward(x, rays, angles)
e = time.time()
print("Time", e - s)

y = y.cpu().numpy()
# print("Error", np.linalg.norm(img - y)/np.linalg.norm(img))

vol_geom = astra.create_vol_geom(128, 128)
proj_geom = astra.create_proj_geom('parallel', 1.0, 128, -angles.cpu().numpy())
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, y_ = astra.create_sino(x.cpu().numpy(), proj_id)

print("My", np.min(y), np.max(y))
print("Astra", y_.shape, np.min(y_), np.max(y_))
print("Error", np.linalg.norm(y_ - y)/np.linalg.norm(y_))


print()
y -= np.min(y)
y /= np.max(y) / 255
y = y.astype(np.uint8)
cv2.imwrite("res.png", y)

y_ -= np.min(y_)
y_ /= np.max(y_) / 255
y_ = y_.astype(np.uint8)
cv2.imwrite("astra_res.png", y_)

# s = 64
#
# lines = np.asarray([
#     [s, 0, 0, 1],
#     [0, s, 1, 0],
#     [-s, 0, 0, 1],
#     [0, -s, 1, 0],
# ], dtype=np.float32)
#

