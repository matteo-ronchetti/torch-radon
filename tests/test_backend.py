from utils import generate_random_images, relative_error
from astra_wrapper import AstraWrapper
from unittest import TestCase
import torch
import numpy as np
from torch_radon import Radon


class TestBackend(TestCase):
    def run_test(self, device, batch_size, image_size, n_angles=180):

        # generate random images
        x = generate_random_images(batch_size, image_size)
        self.assertEqual(x.shape, (batch_size, image_size, image_size))

        angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)

        # astra
        astra = AstraWrapper(angles)

        astra_fp_id, astra_fp = astra.forward(x)
        astra_bp = astra.backproject(astra_fp_id, image_size, batch_size)

        # our implementation
        radon = Radon(image_size).to(device)
        x = torch.FloatTensor(x).to(device)
        angles = torch.FloatTensor(angles).to(device)

        our_fp = radon.forward(x, angles)
        our_bp = radon.backprojection(our_fp, angles)

        forward_error = relative_error(astra_fp, our_fp.cpu().numpy())
        back_error = relative_error(astra_bp, our_bp.cpu().numpy())

        self.assertLess(forward_error, 5e-3)
        self.assertLess(back_error, 5e-3)

        # TODO clear astra

    def test_backend(self):
        device = torch.device('cuda')

        for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
            for image_size in [64, 128, 256, 512]:
                self.run_test(device, batch_size, image_size)
