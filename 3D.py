import torch
import torch_radon as tr
import astra
import numpy as np
import time
import matplotlib.pyplot as plt
import odl
import random

def odl_angles(start, end, n_angles):
    apart = odl.uniform_partition(start, end, n_angles)
    
    correction = abs(end - start) / (2 * n_angles)
    mi = start + correction
    ma = end - correction
    angles = np.linspace(mi, ma, n_angles, endpoint=True)
    
    return apart, angles

def random_3d_volume(volume):
    sz, sy, sx = volume.shape()
    cube = np.zeros(volume.shape(), dtype=np.float32)
    for i in range(25):
        x = random.randint(0, sx)
        y = random.randint(0, sy)
        z = random.randint(0, sz)
        dx = random.randint(1, 32)
        dy = random.randint(1, 32)
        dz = random.randint(1, 32)
        w = random.uniform(0, 1)
        cube[max(z-dz, 0):min(z+dz, sz-1), max(y-dy, 0):min(y+dy, sy-1), max(x-dx, 0):min(x+dx, sx-1)] += w
    return cube

def benchmark_function(f, x, samples, warmup, sync=False):
    for _ in range(warmup):
        f(x)

    if sync:
        torch.cuda.synchronize()
    s = time.time()
    for _ in range(samples):
        f(x)
    if sync:
        torch.cuda.synchronize()
    e = time.time()

    return (e - s) / samples

dtype = np.float32
device = torch.device("cuda")

apart, angles = odl_angles(-np.pi, np.pi, 1024)

volume = tr.Volume3D(256, 256, 256, voxel_size=(1.0, 1.0, 1.0), center=(0, 0, 0))
radon = tr.ConeBeam(256, angles, volume=volume,
                    src_dist=256, det_dist=256, 
                    det_spacing_u=2.0, det_spacing_v=2.0, pitch=0.0, base_z=0.0)

shape = volume.shape()[::-1]
space = odl.uniform_discr(volume.min(), volume.max(), shape, dtype=dtype)
dpart = odl.uniform_partition(
    [-radon.det_count_u/2 * radon.det_spacing_u, -radon.det_count_v/2 * radon.det_spacing_v],
    [radon.det_count_u/2 * radon.det_spacing_u, radon.det_count_v/2 * radon.det_spacing_v],
    [radon.det_count_u, radon.det_count_v]
)

geometry = odl.tomo.ConeBeamGeometry(apart, dpart, radon.src_dist, radon.det_dist, pitch=radon.pitch,
                                     offset_along_axis=radon.base_z)
operator = odl.tomo.RayTransform(space, geometry)

cube = random_3d_volume(volume)

proj_data = operator(cube.transpose(2, 1, 0))
# print(benchmark_function(lambda z: operator(z), cube.transpose(2, 1, 0), 50, 10, True))
print(benchmark_function(lambda z: operator.adjoint(z), proj_data, 15, 5, True))
odl_bp = np.asarray(operator.adjoint(proj_data)).transpose(2, 1, 0)
odl_sino = np.asarray(proj_data).transpose(0, 2, 1)

with torch.no_grad():
    x = torch.FloatTensor(cube).to(device)
    # print(benchmark_function(lambda z: radon.forward(z), x, 50, 10, True))
    y = radon.forward(x)
    print(benchmark_function(lambda z: radon.backward(z), y, 15, 5, True))
    bp = radon.backward(y)

# odl_bp = odl_bp * torch.norm(bp).item() / np.linalg.norm(odl_bp)


# print(torch.sum(y*y), torch.sum(bp*x))
# print(np.sum(odl_sino*odl_sino), np.sum(odl_bp*cube.transpose(2, 1, 0)))


# angle_id = 0
# fig, ax = plt.subplots(1, 3)
# ax = ax.ravel()
# ax[0].imshow(odl_sino[angle_id])
# ax[0].axis("off")
# ax[1].imshow(y[angle_id].cpu().numpy())
# ax[1].axis("off")
# ax[2].imshow(np.abs(odl_sino[angle_id] - y[angle_id].cpu().numpy()))
# ax[2].axis("off")

# slice_id = 46
# fig, ax = plt.subplots(1, 3)
# ax = ax.ravel()
# ax[0].imshow(odl_bp[slice_id])
# ax[0].axis("off")
# ax[1].imshow(bp[slice_id].cpu().numpy())
# ax[1].axis("off")
# ax[2].imshow(np.abs(odl_bp[slice_id] - bp[slice_id].cpu().numpy()))
# ax[2].axis("off")

# plt.show()

# print("Forward", np.linalg.norm(odl_sino - y.cpu().numpy()) / np.linalg.norm(proj_data))
# print("Backward", np.linalg.norm(odl_bp - bp.cpu().numpy()) / np.linalg.norm(odl_bp))
# print(np.linalg.norm(bp.cpu().numpy()), np.linalg.norm(odl_bp))