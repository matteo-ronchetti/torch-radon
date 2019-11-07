import torch
import cv2
import numpy as np
import radon
import time
import astra

def astra_single_fp(x, angles):
    vol_geom = astra.create_vol_geom(128, 128)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 128, -angles.cpu().numpy())
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    x_ = x.cpu().numpy()

    return astra.create_sino(x_, proj_id)

def astra_batch_fp(x, angles):
    x_ = x.cpu().numpy()

    vol_geom = astra.create_vol_geom(x.size(1), x.size(2), x.size(0))
    phantom_id = astra.data3d.create('-vol', vol_geom, data=x_)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, x_.shape[0], x_.shape[1], angles.cpu().numpy())

    return astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)


print(f"Found GPU {torch.cuda.get_device_name(0)}")
device = torch.device('cuda')

# read test image
img = cv2.imread("phantom.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)

# compute rays
s = img.shape[0] // 2
locations = np.arange(2 * s) - s + 0.5
ys = np.sqrt(s ** 2 - locations ** 2)
locations = locations.reshape(-1, 1)
ys = ys.reshape(-1, 1)
rays = np.hstack((locations, -ys, locations, ys))
print("Rays shape", rays.shape)

batch_size = 1024*2
# move to gpu
x = torch.FloatTensor(img).to(device).view(1, 128, 128).repeat(batch_size, 1, 1)
rays = torch.FloatTensor(rays).to(device)
angles = torch.FloatTensor(angles).to(device)

s = time.time()
y = radon.forward(x, rays, angles)
e = time.time()
print("My Time", e - s)
print(y.size())


s = time.time()
_, y_ = astra_batch_fp(x, angles)
e = time.time()
print("Astra Time", e - s)

print("Batch error", np.linalg.norm(y_ - y.cpu().numpy())/np.linalg.norm(y_))

_, ref = astra_single_fp(x[0], angles)
error = np.linalg.norm(ref - y[0].cpu().numpy()) / np.linalg.norm(ref)
print("My-ref Error", error)
error = np.linalg.norm(ref - y_[0]) / np.linalg.norm(ref)
print("Astra-ref Error", error)

y_[0] -= np.min(y_[0])
y_[0] /= np.max(y_[0]) / 255
y_[0] = y_[0].astype(np.uint8)
cv2.imwrite("res.png", y_[0])

ref -= np.min(ref)
ref /= np.max(ref) / 255
ref = ref.astype(np.uint8)
cv2.imwrite("ref.png", ref)