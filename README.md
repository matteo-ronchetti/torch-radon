![Travis (.com)](https://img.shields.io/travis/com/matteo-ronchetti/torch-radon)
[![Documentation Status](https://readthedocs.org/projects/torch-radon/badge/?version=latest)](http://torch-radon.readthedocs.io/?badge=latest)
![GitHub](https://img.shields.io/github/license/matteo-ronchetti/torch-radon)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10GdKHk_6346aR4jl5VjPPAod1gTEsza9)
# TorchRadon: Fast Differentiable Routines for Computed Tomography

TorchRadon is a PyTorch extension written in CUDA that implement differentiable routines
for solving computed tomography (CT) reconstruction problems.
The library is designed to help researchers working on CT problems to combine deep learning
and model-based approaches.

Main features:
 - Forward projection, back projections and shearlet transforms are *differentiable* and
 integrated with PyTorch `.backward()`.
 - Up to 125x *faster* than Astra Toolbox.
 - *Batch operations*: fully exploit the power of modern GPUs by processing multiple images
 in parallel.
 - *Transparent API*: all operations are seamlessly integrated with PyTorch, 
  gradients can  be  computed using `.backward()`,  half precision can be used with Nvidia AMP.
 - *Half precision*: storing data in half precision allows to get sensible speedups 
 when  doing  Radon  forward  and  backward projections with a very small accuracy loss.
 
Implemented operations:
 - Parallel Beam projections
 - Fan Beam projections
 - Shearlet transform
 
 
## Installation
### Precompiled packages
If you are running Linux you can install Torch Radon by running:
```shell script
wget -qO- https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/master/auto_install.py  | python -
```

### Google Colab
You can try the library from your browser using Google Colab, you can find an example
notebook [here](https://colab.research.google.com/drive/10GdKHk_6346aR4jl5VjPPAod1gTEsza9?usp=sharing).

### Docker Image
Docker images with PyTorch CUDA and Torch Radon are available [here](https://hub.docker.com/repository/docker/matteoronchetti/torch-radon).
```shell script
docker pull matteoronchetti/torch-radon
```
To use the GPU in docker you need to use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### Build from source
You need to have [CUDA](https://developer.nvidia.com/cuda-toolkit) and [PyTorch](https://pytorch.org/get-started/locally/) installed, then run:
```shell script
git clone https://github.com/matteo-ronchetti/torch-radon.git
cd torch-radon
python setup.py install
```
If you encounter any problem please contact the author or open an issue.

## Benchmarks
The library is noticeably faster than the Astra Toolbox, especially when data is already on the GPU. Main disadvantage of Astra is that it only takes inputs which are on the CPU, this makes training end-to-end neural networks very inefficient.
The following benchmark compares the speed of Astra Toolbox and Torch Radon:
![V100 Benchmark](pictures/v100.png?raw=true)

If we set `clip_to_circle=True` (consider only the part of the image that is inside the circle) the speed difference is even larger:
![V100 Benchmark circle](pictures/v100_circle.png?raw=true)

These results hold also on a cheap laptop GPU: 
![GTX1650 Benchmark](pictures/gtx1650.png?raw=true)

## Cite
If you are using TorchRadon in your research, please cite the following paper:
```
@misc{torch_radon,
Author = {Matteo Ronchetti},
Title = {TorchRadon: Fast Differentiable Routines for Computed Tomography},
Year = {2020},
Eprint = {arXiv:2009.14788},
}
```

## Testing
Install testing dependencies with `pip install -r test_requirements.txt`
then test with:
```shell script
nosetests tests/
```
