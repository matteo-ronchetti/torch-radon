# Computational Tomography in Pytorch
This library contains fast CUDA implementation of transforms needed for
working with computed tomography data in Pytorch. It allows the training of end-to-end models that takes sinograms as inputs and produce images as output.

Main features:
 - All operations work directly on Pytorch GPU tensors.
 - Forward and back projections are differentiable and integrated with Pytorch `.backward()`.
 - Up to 50x faster than Astra Toolbox.
 - Supports half precision and can used togheter with amp for faster training.
 
## Example
Simple example that uses Pytorch models to filter both the sinogram and the image
```python
import torch
import torch.nn as nn
import numpy as np
from torch_radon import Radon

batch_size = 8
n_angles = 64
image_size = 128

device = torch.device('cuda')

# instantiate Radon transform and loss
radon = Radon(image_size).to(device)
criterion = nn.L1Loss()

# Instantiate a model for the sinogram and one for the image
channels = 4
sino_model = nn.Conv2d(1, channels, 5, padding=2).to(device)
image_model = nn.Conv2d(channels, 1, 3, padding=1).to(device)

# instantiate measuring angles
angles = torch.FloatTensor(np.linspace(0, np.pi, n_angles)).to(device)

# create empty images
x = torch.FloatTensor(batch_size, 1, image_size, image_size).to(device)

# forward projection
sinogram = radon.forward(x, angles)

# apply sino_model to sinograms
sinogram = sino_model(sinogram)

# back projection
backprojected = radon.backprojection(sinogram, angles)

# apply image_model to backprojected images
y = image_model(backprojected)

# compute image reconstruction error
loss = criterion(y, x)

# backward works as usual
loss.backward()
```

## Installation
### Precompiled packages
TODO
### Build from source
You need to have CUDA and Pytorch installed, then run:
```shell script
git clone https://github.com/matteo-ronchetti/torch-radon.git
cd torch-radon
make install
```

## Half Precision
Storing input data in half precision (float16) makes forward and back projection operations a lot (2x-2.6x times) faster.
Because internally operations are done in single precision the numerical error is small. TODO show numerical error.


## Benchmarks
The library is noticeably faster than the Astra Toolbox when considering also CPU-GPU-CPU copies (first two columns) and is a lot faster when processing data which is already on the GPU (last two columns). Main disadvantage of Astra is that it only takes inputs which are on the CPU, this makes training end-to-end neural networks very inefficient.
![V100 Benchmark](V100.png?raw=true)
![GTX1650 Benchmark](gtx1650.png?raw=true)
Note: batch size with Astra is achieved by using a 3D parallel projection.


## Testing
Install testing dependencies with `pip install -r test_requirements.txt`
then test with:
```shell script
nosetests tests/
```
