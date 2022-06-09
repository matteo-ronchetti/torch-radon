import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torch_radon


def xray_to_image(orig):
    x = orig.copy()
    x -= np.min(x)
    x /= np.max(x) / 255
    x = np.rint(x).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)


# ct = np.load("/home/matteo/ImFusion/learnedposecost/data/ct_0.npy")
ct_point = np.asarray([250, 50, 150])
ct = np.zeros((440, 227, 350), dtype=np.float32)
ct[ct_point[2]-30:ct_point[2]+30, ct_point[1]-30:ct_point[1]+30, ct_point[0]-30:ct_point[0]+30] = 1


with torch.no_grad():
    tct = torch.FloatTensor(ct).unsqueeze(0)
    tct = tct.cuda()

    tex_cache = torch_radon.cuda_backend.TextureCache(8)
    spacing = 5.0
    dist = 200.0

    proj = torch_radon.cuda_backend.Projection3D.ConeBeam(256, 256, dist, dist, spacing, spacing, 0.0)
    proj.setPose(0.3, -0.2, 0.0, 45, -20, 5)

    angles = [0, 0]
    tangles = torch.FloatTensor(angles).cuda()
    vol_cfg = torch_radon.cuda_backend.VolumeCfg(*tct.size()[1:], 0.5, 0.5, 2.0)
    exec_cfg = torch_radon.cuda_backend.ExecCfg(8, 16, 8, 1)
    xray = torch_radon.cuda_backend.forward_batch(tct, tangles, tex_cache, vol_cfg, [
                                                  proj, proj], exec_cfg)[0, 0].cpu().numpy()
    print(xray.shape)
    M = torch_radon.cuda_backend.projection_matrices(angles, vol_cfg, [proj, proj])[0]
    print(M.shape)
    print(M)

    p = ct_point.copy().astype(np.float32)
    print(p.shape, M.shape)
    q = p @ M[:3, :3].T + M[:3, 3]
    q /= q[2]
    print(q)
    q = [q[0] + 128, 128 + q[1]]
    # q = q[:2] + 128
    print(q)


img = xray_to_image(xray)
cv2.circle(img, (int(q[0]), int(q[1])), 2, (0, 0, 255), -1)
plt.imshow(img)
plt.show()

#
# ct[ct_point[2]-30:ct_point[2]+30, ct_point[1]-30:ct_point[1]+30, ct_point[0]-30:ct_point[0]+30] = 1

# mpr = xray_to_image(ct[int(ct_point[2])])
# cv2.circle(mpr, (int(ct_point[0]), int(ct_point[1])), 4, (0, 0, 255), -1)
# plt.imshow(mpr)
