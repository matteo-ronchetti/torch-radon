import sys

base_url = "https://rosh-public.s3-eu-west-1.amazonaws.com/radon"

assert sys.platform == "linux", "ERROR  precompiled packages are only available for Linux"

python_version = sys.version_info.major * 10 + sys.version_info.minor
assert python_version in [36, 37], "ERROR precompiled packages are only available for python 3.6 and 3.7"

python_version = f"cp{python_version}-cp{python_version}m"

try:
    import torch

    torch_version = torch.__version__[:3].replace(".", "")
    print("To install Torch Radon run the following command:")
    print(f"pip install {base_url}/{torch_version}/torch_radon-0.0.1-{python_version}-linux_x86_64.whl")

except:
    print("ERROR you need to have PyTorch installed")
