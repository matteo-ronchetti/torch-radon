import numpy as np
import torch_radon as tr
from unittest import TestCase


def random_pose():
    R = np.random.randn(3, 3)
    U, _, V = np.linalg.svd(R)
    R = U @ V.T
    P = np.eye(4, 4, dtype=np.float32)
    P[: 3, :3] = R
    P[:3, -1] = np.random.randn(3) * 5
    return P


class TestMatrices(TestCase):
    def test_matrices(self):
        p = tr.cuda_backend.Projection3D.ConeBeam(128, 128, 20, 20, 1.0, 1.0, 0.0)

        self.assertLess(np.linalg.norm(np.eye(4) - p.imageToWorld), 1e-6)

        pose = random_pose()
        p.imageToWorld = pose
        self.assertLess(np.linalg.norm(pose - p.imageToWorld), 1e-6)

        v = tr.cuda_backend.VolumeCfg(100, 100, 100, 1, 1, 1)
        p.updateMatrices(v)
        self.assertLess(np.linalg.norm(np.linalg.inv(pose) - p.worldToImage), 1e-5)
