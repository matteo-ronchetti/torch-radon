from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='torch_radon',
      version="0.0.1",
      author="Matteo Ronchetti",
      author_email="mttronchetti@gmail.com",
      description="Radon transform in Pytorch",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/matteo-ronchetti/torch-radon",

      packages=['torch_radon'],
      package_dir={'torch_radon': './torch_radon'},
      ext_modules=[
          CUDAExtension('torch_radon_cuda', ['src/pytorch.cpp'],
                        include_dirs=[os.path.abspath('include')],
                        library_dirs=[os.path.abspath("objs/cuda")],
                        libraries=["radon"],
                        # strip debug symbols
                        extra_link_args=['-Wl,--strip-all']
                        )
      ],
      cmdclass={'build_ext': BuildExtension},
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
      ]
      )
