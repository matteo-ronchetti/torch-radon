"""Since this package is a pytorch extension, this setup file uses the custom
CUDAExtension build system from pytorch. This ensures that compatible compiler
args, headers, etc for pytorch.

Read more at the pytorch docs:
https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension
"""
import os.path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch_radon",
    version="2.0.0",
    author="Matteo Ronchetti",
    author_email="mttronchetti@gmail.com",
    description="Radon transform in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matteo-ronchetti/torch-radon",
    packages=["torch_radon"],
    package_dir={
        "": "src/python",
    },
    ext_modules=[
        CUDAExtension(
            name="torch_radon_cuda",
            sources=[
                "src/backprojection.cu",
                "src/fft.cu",
                "src/forward.cu",
                "src/log.cpp",
                "src/noise.cu",
                "src/parameter_classes.cu",
                "src/pytorch.cpp",
                "src/symbolic.cpp",
                "src/texture.cu",
            ],
            include_dirs=[os.path.abspath("include")],
            extra_compile_args={
                "cxx": [
                    # GNU++14 required for hexfloat extension used in rmath.h
                    "-std=gnu++14",
                ],
                "nvcc": [
                    # __half conversions required in backprojection
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            }),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    install_requires=[
        "torch",
        "scipy",
        "numpy",
        "alpha-transform",
    ],
    extras_require={
        "testing": [
            "astra-toolbox",
            "dxchange",
            "matplotlib",
            "nose",
            "numpy",
            "parameterized",
            "scikit-image",
        ]
    },
)
