from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='torch_radon',
      py_modules=['torch_radon'],
      ext_modules=[
          CUDAExtension('torch_radon_cuda', [
              'radon.cpp',
              'radon_cuda.cu',
          ])
      ],
      cmdclass={'build_ext': BuildExtension})
