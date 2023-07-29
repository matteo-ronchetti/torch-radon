![Travis (.com)](https://img.shields.io/travis/com/matteo-ronchetti/torch-radon)
[![Documentation Status](https://readthedocs.org/projects/torch-radon/badge/?version=latest)](http://torch-radon.readthedocs.io/?badge=latest)
![GitHub](https://img.shields.io/github/license/matteo-ronchetti/torch-radon)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A8axh4TMn8C7v4velMgDeovncDKDUXf8?usp=sharing)



# torch-radon (forked)

TorchRadon is a PyTorch extension written in CUDA that implements differentiable routines
for solving computed tomography (CT) reconstruction problems.

The library is designed to help researchers working on CT problems to combine deep learning
and model-based approaches.

Main features:
 - Forward projections, back projections and shearlet transforms are **differentiable** and
 integrated with PyTorch `.backward()`.
 - Up to 125x **faster** than Astra Toolbox.
 - **Batch operations**: fully exploit the power of modern GPUs by processing multiple images
 in parallel.
 - **Transparent API**: all operations are seamlessly integrated with PyTorch, 
  gradients can  be  computed using `.backward()`,  half precision can be used with Nvidia AMP.
 - **Half precision**: storing data in half precision allows to get sensible speedups 
 when  doing  Radon  forward  and  backward projections with a very small accuracy loss.
 
Implemented operations:
 - Parallel Beam projections
 - Fan Beam projections
 - Shearlet transform
 
## Google Colab

You can try the library from your browser using Google Colab, you can find an example
notebook [here](https://colab.research.google.com/drive/1A8axh4TMn8C7v4velMgDeovncDKDUXf8?usp=sharing).
If you are using this repository in Google Colab, follow these commands to build from source and install the modified version:

```
!git clone https://github.com/sypsyp97/torch-radon.git
!wget https://github.com/sypsyp97/torch-radon/raw/master/examples/phantom.npy
%cd torch-radon
!pip install .
%cd ..
```

## Acknowledgement

This is a fork of the [torch-radon](https://github.com/matteo-ronchetti/torch-radon.git) project originally created and maintained by [Matteo Ronchetti](https://github.com/matteo-ronchetti). 

## TODO List

- [x] `torch.rfft` and `torch.irfft` are not supported any more, move to `torch.fft.rfft` and `torch.fft.irfft`
- [ ] Extend to Cone Beam projections

## Cite

If you are using TorchRadon in your research, please cite the following paper:
```
@article{torch_radon,
Author = {Matteo Ronchetti},
Title = {TorchRadon: Fast Differentiable Routines for Computed Tomography},
Year = {2020},
Eprint = {arXiv:2009.14788},
journal={arXiv preprint arXiv:2009.14788},
}
