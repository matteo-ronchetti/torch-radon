from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

setup(name='torch_radon',
      py_modules=['torch_radon'],
      ext_modules=[
          CUDAExtension('torch_radon_cuda', ['src/radon_pytorch.cpp'],
                        include_dirs=['include'],
                        library_dirs=["objs/cuda"],
                        libraries=["radon"]
                        )
      ],
      cmdclass={'build_ext': BuildExtension})
