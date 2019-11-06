from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='radon',
      ext_modules=[
          CUDAExtension('lltm_cuda', [
              'radon.cpp',
              'radon_cuda.cu',
          ])
      ],
      cmdclass={'build_ext': BuildExtension})
