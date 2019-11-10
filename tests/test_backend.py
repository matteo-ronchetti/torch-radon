from utils import generate_random_images, relative_error
from astra_wrapper import AstraWrapper
from nose.tools import assert_less
import torch
import numpy as np
from torch_radon import Radon
from parameterized import parameterized


device = torch.device('cuda')

params = []
for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
    for image_size in [64, 128, 256, 512]:
        params.append((device, batch_size, image_size))
        
@parameterized(params)
def test_error(device, batch_size, image_size, n_angles=180):
    # generate random images
    x = generate_random_images(batch_size, image_size)

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

    # TODO clear astra
    
    print(batch_size, image_size, forward_error, back_error)
    assert_less(forward_error, 1e-2)
    assert_less(back_error, 5e-3)
    
# class TestBackend(TestCase):
#     def test_backend(self):
#         device = torch.device('cuda')

#         for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
#             for image_size in [64, 128, 256, 512]:
#                 forward_error, back_error = compute_error(device, batch_size, image_size)
#                 print(batch_size, image_size, forward_error, back_error)
#                 self.assertLess(forward_error, 1)
#                 self.assertLess(back_error, 1)