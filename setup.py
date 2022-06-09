from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from make import build

with open("README.md", "r") as fh:
    long_description = fh.read()

cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
print(f"Using CUDA_HOME={cuda_home}")
build(cuda_home=cuda_home)

setup(name='torch_radon',
      version="2.0.0",
      author="Matteo Ronchetti",
      author_email="mttronchetti@gmail.com",
      description="Radon transform in PyTorch",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/matteo-ronchetti/torch-radon",

      packages=['torch_radon'],
      package_dir={
          'torch_radon': './torch_radon',
      },
      ext_modules=[
          #   CUDAExtension('torch_radon_cuda', [os.path.abspath('src/pytorch.cpp')],
          #                 include_dirs=[os.path.abspath('include')],
          #                 library_dirs=[os.path.abspath("objs")],
          #                 libraries=["m", "c", "gcc", "stdc++", "cufft", "torchradon"],
          #                 # strip debug symbols
          #                 extra_link_args=["-Wl,--strip-all"]
          #                 ),
          Pybind11Extension("tr", [os.path.abspath("src/python.cpp")],   include_dirs=[os.path.abspath('include')],
                            library_dirs=[os.path.abspath("objs")],
                            libraries=["torchradon"],
                            # strip debug symbols
                            extra_link_args=["-Wl,--strip-all"], cxx_std=14)
      ],
      cmdclass={'build_ext': BuildExtension},
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
      ],
      install_requires=[
          "scipy",
          "alpha-transform"
      ],
      )
