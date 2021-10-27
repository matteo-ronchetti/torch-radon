"""Demonstrates integrating the torch_radon package into a Pytorch model.

Requires the tooth dataset from tomopy and the packages listed in the imports.
"""

import os
from typing import Union
import unittest

import dxchange
import numpy as np
import torch
import torch_radon

assert torch_radon.__version__ == "2.0"


class ParallelRadonModule(torch_radon.ParallelBeam, torch.nn.Module):
    def __init__(
        self,
        det_count: int,
        angles: Union[list, np.array, torch.Tensor, tuple],
        volume: int,
        det_spacing: float = 1.0,
    ):
        torch.nn.Module.__init__(self)
        torch_radon.ParallelBeam.__init__(
            self,
            det_count=det_count,
            angles=angles,
            det_spacing=det_spacing,
            volume=volume,
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(volume, volume, dtype=torch.float32))

    def forward(self):
        return torch_radon.ParallelBeam.forward(self, self.weight)


class TestParallelRadonModule(unittest.TestCase):
    """Tests the ParallelRadonModule by reconstructing the tooth dataset."""
    def setUp(self):
        # Load the tooth dataset
        self.proj, flat, dark, self.theta = dxchange.read_aps_32id(
            fname=os.path.join(os.path.dirname(__file__), 'tooth.h5'),
            sino=(0, 1),  # only the first slice
        )
        # flatfield correction
        self.proj = (self.proj - dark[0]) / (flat[0] - dark[0])
        # linearize the problem
        self.proj = -np.log(self.proj)[:, 0]
        # Move the rotation center to the center of projections
        center = 295.890625
        self.proj = np.roll(
            self.proj,
            -int(center - self.proj.shape[-1] // 2),
            axis=-1,
        )

    def test_parallel_radon_module(self, num_epoch=256):

        assert torch.cuda.is_available(), "Module is CUDA only!"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        theta = torch.from_numpy(self.theta).type(torch.float32).to(device)
        data = torch.from_numpy(self.proj).type(torch.float32).to(device)
        var = torch.ones(
            data.shape,
            dtype=torch.float32,
            requires_grad=True,
        ).to(device)

        model = ParallelRadonModule(
            det_count=data.shape[-1],
            angles=theta,
            volume=data.shape[-1],
        ).to(device)

        lossf = torch.nn.GaussianNLLLoss().to(device)

        optimizer = torch.optim.Adam(model.parameters())

        loss_log = []
        for epoch in range(num_epoch):
            pred = model()
            loss = lossf(pred, data, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            print(f"loss: {loss_log[-1]:.3e}  [{epoch:>5d}/{num_epoch:>5d}]")

        obj = model.weight.cpu().detach().numpy()

        _save_lamino_result({'obj': obj, 'costs': loss_log}, 'torch')


def _save_lamino_result(result, algorithm):
    try:
        import matplotlib.pyplot as plt
        fname = os.path.join(os.path.dirname(__file__), 'tooth')
        os.makedirs(fname, exist_ok=True)
        plt.figure()
        plt.title(algorithm)
        plt.plot(result['costs'])
        plt.semilogy()
        plt.savefig(os.path.join(fname, 'convergence.svg'))
        slice_id = int(35 / 128 * result['obj'].shape[0])
        plt.imsave(
            f'{fname}/{slice_id}-tooth.png',
            result['obj'].astype('float32'),
        )
        import skimage.io
        skimage.io.imsave(
            f'{fname}/tooth.tiff',
            result['obj'].astype('float32'),
        )

    except ImportError:
        pass
