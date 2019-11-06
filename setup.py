from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='radon',
      ext_modules=[cpp_extension.CppExtension('radon', ['radon.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
