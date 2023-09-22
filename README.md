`carterbox/torch-radon` is a fork of `matteo-ronchetti/torch-radon` with some
modules removed (shearlets, reconstruction) and the build system replaced with
the extension system from PyTorch. This fork is maintained separately because
the upstream project is unmaintained. If the upstream project becomes active
again, this fork will attempt to merge its improvements upstream.

# TorchRadon: Fast Differentiable Routines for Computed Tomography

TorchRadon is a PyTorch extension written in CUDA that implements
differentiable routines for solving computed tomography (CT) reconstruction
problems.

The library is designed to help researchers working on CT problems to combine
deep learning and model-based approaches.

Main features:
 - Forward projections, back projections and shearlet transforms are
 **differentiable** and integrated with PyTorch `.backward()` .
 - Up to 125x **faster** than Astra Toolbox.
 - **Batch operations**: fully exploit the power of modern GPUs by processing
 multiple images in parallel.
 - **Transparent API**: all operations are seamlessly integrated with PyTorch,
  gradients can  be  computed using `.backward()` , half precision can be used
  with Nvidia AMP.
 - **Half precision**: storing data in half precision allows to get sensible
 speedups when  doing  Radon  forward  and  backward projections with a very
 small accuracy loss.

Implemented operations:
 - Parallel Beam projections
 - Fan Beam projections
 - 3D Conebeam projection

## Speed

TorchRadon is much faster than competing libraries:

![benchmark](https://raw.githubusercontent.com/matteo-ronchetti/tomography-benchmarks/master/figures/tesla_t4_barplot.png)

See the [Tomography Benchmarks
repository](https://github.com/matteo-ronchetti/tomography-benchmarks) for more
detailed benchmarks.

## Installation

Currently only Linux is supported.

## Install via the Conda package manager and conda-forge channel

Please read about how to setup and use the conda package manager before attempting the following command.

```bash
conda install --channel conda-forge carterbox-torch-radon
```

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

```

## Testing

Install testing dependencies with `pip install .[testing]`
then test with:
```shell script
pytest tests/
```
