import torch
import cv2
import numpy as np
import radon
import time
import astra

print(f"Found GPU {torch.cuda.get_device_name(0)}")
device = torch.device('cuda')

# read test image
img = cv2.imread("phantom.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)

# compute rays
s = img.shape[0] // 2
locations = np.arange(2 * s) - s + 0.5
print(locations[0], locations[-1])
ys = np.sqrt(s ** 2 - locations ** 2)
locations = locations.reshape(-1, 1)
ys = ys.reshape(-1, 1)
rays = np.hstack((locations, -ys, locations, ys))
print("Rays shape", rays.shape)

# move to gpu
x = torch.FloatTensor(img).to(device).view(1, 128, 128).repeat(10, 1, 1)
rays = torch.FloatTensor(rays).to(device)
angles = torch.FloatTensor(angles).to(device)

# s = time.time()
# for i in range(1000):
y = radon.forward(x, rays, angles)
# e = time.time()
# print("Time", e - s)
print(y.size())
y = y.cpu().numpy()[9]
# print("Error", np.linalg.norm(img - y)/np.linalg.norm(img))

vol_geom = astra.create_vol_geom(128, 128)
proj_geom = astra.create_proj_geom('parallel', 1.0, 128, -angles.cpu().numpy())
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
x_ = x.cpu().numpy()[0]

# s = time.time()
# for i in range(1000):
sinogram_id, y_ = astra.create_sino(x_, proj_id)
# e = time.time()
# print("Astra Time", e - s)

print("My", y.shape, np.min(y), np.max(y))
print("Astra", y_.shape, np.min(y_), np.max(y_))
print("Error", np.linalg.norm(y_ - y) / np.linalg.norm(y_))

y -= np.min(y)
y /= np.max(y) / 255
y = y.astype(np.uint8)
cv2.imwrite("res.png", y)

y_ -= np.min(y_)
y_ /= np.max(y_) / 255
y_ = y_.astype(np.uint8)
cv2.imwrite("astra_res.png", y_)
