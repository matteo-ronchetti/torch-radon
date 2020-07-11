import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time

from torch_radon import Radon
from torch_radon.shearlet import Shearlet
from torch_radon.solvers import Landweber
from utils import show_images


def CG(radon, a, b, x, y, max_iter=2000):
    Ax = lambda z: a * radon.backprojection(radon.forward(z), extend=False) + b * z
    r = y - Ax(x)
    p = r.clone()
    r_n = torch.sum(r ** 2).item()

    values = []
    for i in range(max_iter):
        Ap = Ax(p)
        alpha = r_n / torch.sum(p * Ap).item()
        x += alpha * p
        r_next = r - alpha * Ap

        r_next_n = torch.sum(r_next ** 2).item()
        if r_next_n < 1e-4:
            break
        values.append(r_next_n)  # torch.norm(y - Ax(x)).item())

        beta = r_next_n / r_n
        r_n = r_next_n
        p = r_next + beta * p
        r = r_next.clone()

    # plt.plot(values)
    # plt.yscale("log")
    # plt.show()

    return x, values


def shrink(a, b):
    return (torch.abs(a) - b).clamp_min(0) * torch.sign(a)


batch_size = 1
n_angles = 512 // 4
image_size = 512

img = np.load("phantom.npy")
device = torch.device('cuda')

# instantiate Radon transform
angles = np.linspace(0, np.pi / 4, n_angles, endpoint=False)
radon = Radon(image_size, angles)
shearlet = Shearlet(512, 512, [0.5] * 5, cache=None)  # ".cache")

with torch.no_grad():
    x = torch.FloatTensor(img).reshape(1, image_size, image_size).to(device)
    sinogram = radon.forward(x)
    bp = radon.backward(sinogram, extend=False)

    # f, values = CG(radon, 1.0 / 512**2, 0.0001, bp.clone(), bp)
    #
    # print(torch.norm(x - f)/torch.norm(x))
    sc = shearlet.forward(bp)
    p_0 = 0.02
    p_1 = 0.1
    w = 3 ** shearlet.scales / 400
    w = w.view(1, -1, 1, 1).to(device)

    u_2 = torch.zeros_like(bp)
    z_2 = torch.zeros_like(bp)
    u_1 = torch.zeros_like(sc)
    z_1 = torch.zeros_like(sc)
    f = torch.zeros_like(bp)

    s = time.time()
    for i in range(50):
        cg_y = p_0 * bp + p_1 * shearlet.backward(z_1 - u_1) + (z_2 - u_2)
        f, values = CG(radon, p_0, 1 + p_1, f.clone(), cg_y)
        sh_f = shearlet.forward(f)

        print(torch.norm(radon.forward(f) - sinogram).item() ** 2, torch.sum(torch.abs(sh_f)).item())

        z_1 = shrink(sh_f + u_1, p_0 / p_1 * w)
        z_2 = (f + u_2).clamp_min(0)
        u_1 = u_1 + sh_f - z_1
        u_2 = u_2 + f - z_2
    e = time.time()
    print(e - s)

    sc = shearlet.forward(f)
    sc_x = shearlet.forward(x)
    vs = []
    for i in range(59):
        vs.append(torch.norm(sc[0, i]).item() / torch.norm(sc_x[0, i]).item())

    plt.plot(vs)
    plt.show()

    plt.imshow(f[0].cpu())
    plt.show()

    plt.plot(values)
    plt.yscale("log")
    plt.show()
