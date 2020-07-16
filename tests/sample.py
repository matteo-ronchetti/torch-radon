from torch_radon import Radon
import torch
import matplotlib.pyplot as plt
import numpy as np
import astra


class AstraWrapper:
    def __init__(self, angles):
        self.angles = angles

        self.projectors = []
        self.algorithms = []
        self.data2d = []
        self.data3d = []

    def forward(self, x):
        vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2], x.shape[0])
        phantom_id = astra.data3d.create('-vol', vol_geom, data=x)
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, x.shape[0], x.shape[1], -self.angles)

        proj_id, y = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

        self.projectors.append(proj_id)
        self.data3d.append(phantom_id)

        return proj_id, y

    def backproject(self, proj_id, s, bs):
        vol_geom = astra.create_vol_geom(s, s, bs)
        rec_id = astra.data3d.create('-vol', vol_geom)

        # Set up the parameters for a reconstruction algorithm using the GPU
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)

        self.algorithms.append(alg_id)
        self.data3d.append(rec_id)

        return astra.data3d.get(rec_id)

    def forward_single(self, x):
        vol_geom = astra.create_vol_geom(x.shape[0], x.shape[1])
        proj_geom = astra.create_proj_geom('parallel', 1.0, x.shape[0], self.angles)
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        self.projectors.append(proj_id)

        return astra.create_sino(x, proj_id)

    def fbp(self, x):
        s = x.shape[0]
        proj_id, _ = self.forward_single(x)
        vol_geom = astra.create_vol_geom(s, s)
        rec_id = astra.data2d.create('-vol', vol_geom)

        # create configuration
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        self.projectors.append(proj_id)
        self.algorithms.append(alg_id)
        self.data2d.append(rec_id)

        return astra.data2d.get(rec_id)

    def clean(self):
        # clean all astra stuff
        for pid in self.projectors:
            astra.projector.delete(pid)

        for pid in self.algorithms:
            astra.algorithm.delete(pid)

        for pid in self.data2d:
            astra.data2d.delete(pid)

        for pid in self.data3d:
            astra.data3d.delete(pid)

    def __del__(self):
        self.clean()


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / np.linalg.norm(ref)


def circle_mask(size):
    radius = (size - 1) / 2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - radius) ** 2 + (c1 - radius) ** 2) <= (radius) ** 2


def generate_random_images(n, size, masked=False):
    # generate images
    batch = np.random.uniform(0.0, 1.0, (n, size, size)).astype(np.float32)

    if masked:
        # create and apply circular mask
        mask = circle_mask(size)
        batch *= mask

    return batch


# import math
#
# rsx = 3
# rsy = -1
# rdy = 0.8
# rdx = -1
# v = 2.0
#
# a = rdx * rdx + rdy * rdy
# b = rsx * rdx + rsy * rdy
# c = rsx * rsx + rsy * rsy - v * v
#
# print(a, b, c)
#
# # min_clip to 1 to avoid getting empty rays
# delta_sqrt = math.sqrt(max(b * b - a * c, 1.0))
# alpha_e = (-b - delta_sqrt) / a
# alpha_s = (-b + delta_sqrt) / a
#
# rsx += rdx * alpha_s
# rsy += rdy * alpha_s
# rdx *= (alpha_e - alpha_s)
# rdy *= (alpha_e - alpha_s)
#
# print(rsx, rsy, rsx**2 + rsy**2 - v**2)
# print(rdx, rdy, (rsx+rdx)**2 + (rsy+rdy)**2 - v**2)

device = torch.device('cuda')

angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)

batch_size = 4
image_size = 256
astraw = AstraWrapper(angles)

x = generate_random_images(batch_size, image_size, masked=True)

astra_fp_id, astra_fp = astraw.forward(x)

# our implementation
radon = Radon(image_size, angles, clip_to_circle=True)
x = torch.FloatTensor(x).to(device)

our_fp = radon.forward(x)

plt.imshow(astra_fp[0])
plt.figure()
plt.imshow(our_fp[0].cpu().numpy())
plt.show()

print(relative_error(astra_fp, our_fp.cpu().numpy()))
