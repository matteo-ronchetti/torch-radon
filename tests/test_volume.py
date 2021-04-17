import torch
import numpy as np
import torch_radon as tr
from unittest import TestCase
from .utils import generate_random_images


class TestVolume(TestCase):
    def test_non_square(self):
        width = 128
        height = 256
        n_angles = 128
        device = torch.device("cuda")

        img = np.random.uniform(0.0, 1.0, (height, width)).astype(np.float32)
        image_size = max(width, height)
        pad_y = (image_size - height) // 2
        pad_x = (image_size - width) // 2
        padded = np.pad(img, ((pad_y, pad_y), (pad_x,pad_x)))

        angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        #projection = Projection.parallel_beam(image_size)
        projection = Projection.fanbeam(image_size, image_size, image_size)

        radon = Radon(angles, img.shape, projection)
        radon_p = Radon(angles, padded.shape, projection)

        with torch.no_grad():
            x = torch.FloatTensor(img).to(device).to(device) #.half()
            xp = torch.FloatTensor(padded).to(device).to(device) #.half()

            sinogram = radon.forward(x)
            sinogram_p = radon_p.forward(xp)

            bp = radon.backprojection(sinogram)
            bp_p = radon_p.backprojection(sinogram_p)

        self.assertLess(torch.norm(sinogram_p - sinogram) / torch.norm(sinogram), 6e-3)
        bp_p_cropped = bp_p[pad_y:pad_y+height, pad_x:pad_x+width]
        self.assertLess(torch.norm(bp_p_cropped - bp) / torch.norm(bp), 6e-4)


    def test_translation(self):
        dy = 16
        dx = 32
        device = torch.device("cuda")
        
        img = np.zeros((256, 256), dtype=np.float32)
        img[128-16+dy:128+16+dy, 128-16+dx:128+16+dx] = 1.0
        vol = Volume.create_2d(*img.shape)

        img_d = np.zeros((256, 256), dtype=np.float32)
        img_d[128-16:128+16, 128-16:128+16] = 1.0
        vol_d = Volume.create_2d(*img_d.shape, dy, dx)

        # compute croppings that align the images
        cropped_h = img.shape[0] - abs(dy)
        cropped_w = img.shape[1] - abs(dx)
        cropped_img = img[max(dy, 0):max(dy, 0) + cropped_h, max(dx, 0):max(dx, 0) + cropped_w] 
        cropped_img_d = img_d[max(-dy, 0):max(-dy, 0) + cropped_h, max(-dx, 0):max(-dx, 0) + cropped_w]
        # if this fails then there is an error in cropping
        self.assertLess(np.linalg.norm(cropped_img-cropped_img_d), 1e-6)

        angles = np.linspace(0, np.pi, 128, endpoint=False)
        projection = Projection.parallel_beam(256)

        radon = Radon(angles, vol, projection)
        radon_d = Radon(angles, vol_d, projection)

        with torch.no_grad():
            x = torch.FloatTensor(img).to(device)
            x_d = torch.FloatTensor(img_d).to(device)

            sinogram = radon.forward(x)
            sinogram_d = radon_d.forward(x_d)

            bp = radon.backprojection(sinogram)
            bp_d = radon_d.backprojection(sinogram_d)

        self.assertLess(torch.norm(sinogram_d - sinogram) / torch.norm(sinogram), 2e-3)

        cropped_bp = bp[max(dy, 0):max(dy, 0) + cropped_h, max(dx, 0):max(dx, 0) + cropped_w] 
        cropped_bp_d = bp_d[max(-dy, 0):max(-dy, 0) + cropped_h, max(-dx, 0):max(-dx, 0) + cropped_w]
        self.assertLess(torch.norm(cropped_bp - cropped_bp_d) / torch.norm(cropped_bp), 2e-4)

    def test_scale(self):
        sy = 1
        sx = 2
        device = torch.device("cuda")
        
        img = np.zeros((256, 256), dtype=np.float32)
        img[128-16:128+16, 128-16:128+16] = 1.0
        vol = Volume.create_2d(*img.shape)

        img_d = img[::sy, ::sx]
        vol_d = Volume.create_2d(*img_d.shape, sy=sy, sx=sx)

        angles = np.linspace(0, np.pi, 128, endpoint=False)
        projection = Projection.parallel_beam(256)

        radon = Radon(angles, vol, projection)
        radon_d = Radon(angles, vol_d, projection)

        with torch.no_grad():
            x = torch.FloatTensor(img).to(device)
            x_d = torch.FloatTensor(img_d).to(device)

            sinogram = radon.forward(x)
            sinogram_d = radon_d.forward(x_d)

            bp = radon.backprojection(sinogram)
            bp_d = radon_d.backprojection(sinogram_d)

        self.assertLess(torch.norm(sinogram_d - sinogram) / torch.norm(sinogram), 2e-3)

        # cropped_bp = bp[max(dy, 0):max(dy, 0) + cropped_h, max(dx, 0):max(dx, 0) + cropped_w] 
        # cropped_bp_d = bp_d[max(-dy, 0):max(-dy, 0) + cropped_h, max(-dx, 0):max(-dx, 0) + cropped_w]
        # self.assertLess(torch.norm(cropped_bp - cropped_bp_d) / torch.norm(cropped_bp), 2e-4)