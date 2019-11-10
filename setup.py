from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


setup(name='torch_radon',
      py_modules=['torch_radon'],
      ext_modules=[
          CUDAExtension('torch_radon_cuda', [
              'src/radon.cpp',
              'src/radon_cuda.cu',
          ],
          include_dirs=['include'])
          #extra_compile_args={"cxx": [], "nvcc": ["-I" + os.path.join(dir_path, "include")]})
      ],
      cmdclass={'build_ext': BuildExtension})
