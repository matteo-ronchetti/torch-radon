from skbuild import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torch_radon',
    version="2.0.0",
    author="Matteo Ronchetti",
    author_email="mttronchetti@gmail.com",
    description="Radon transform in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matteo-ronchetti/torch-radon",
    # NOTE: Empty package name picks up torch_radon_cuda.so
    packages=['torch_radon', ''],
    package_dir={
        '': 'src/python',
    },
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    install_requires=[
        "scipy",
        "alpha-transform",
    ],
)
