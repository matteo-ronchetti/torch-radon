from torch_radon import Radon, RadonFanbeam
import torch
import numpy as np
import matplotlib.pyplot as plt

x = np.load("examples/phantom.npy")
print(x.shape)
bs = 12

x = torch.FloatTensor(x).cuda().view(1, 512, 512)
for i in range(bs):
    z = torch.cat([x] * bs, dim=0)
    # z[i:] *= 0

    angles = np.linspace(0, np.pi, num=512, endpoint=False)
    radon = Radon(512, angles, det_count=580, clip_to_circle=True)

    y = radon.forward(z)
    # filtered = radon.filter_sinogram(y)
    fbp = radon.backward(y)

    torch.cuda.synchronize()

    print(i, np.linalg.norm(z.cpu() - fbp.cpu()) / (np.linalg.norm(z.cpu() + 1e-5)))

    for i in range(bs):
        print(i)
        plt.imshow(fbp[i].cpu())
        plt.show()
