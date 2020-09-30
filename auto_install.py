import os
import sys

version = "0.0.1"
base_url = "https://rosh-public.s3-eu-west-1.amazonaws.com/radon"

green = "\u001b[32m"
red = "\u001b[31m"
blue = "\u001b[34m"
reset = "\u001b[0m"


ok = f"{green}OK{reset}"
error = f"{red}ERROR{reset}"
not_found = f"{red}Not found{reset}"

print(f"{blue}Checking requirements{reset}")

# OS
platform = sys.platform
print("Operating System:", platform, end=" ")
if platform == "linux":
    print(ok)
else:
    print(error)
    print("Precompiled packages are only available for Linux")
    sys.exit(1)

# Python
python_version = sys.version_info.major * 10 + sys.version_info.minor
python_version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
print("Python version:", python_version_str, end=" ")
if python_version in [36, 37]:
    print(ok)
else:
    print(error)
    print("Precompiled packages are only available for Python 3.6 and 3.7")
    sys.exit(1)

# PyTorch
try:
    import torch

    torch_available = True
except:
    torch_available = False

print("PyTorch:", end=" ")
if torch_available:
    torch_version = torch.__version__[:3]
    print(torch_version, end=" ")
    if torch_version in ["1.6", "1.5", "1.4", "1.3"]:
        print(ok)
    else:
        print(error)
        print("Precompiled packages are build for PyTorch 1.3, 1.4, 1.5 and 1.6")
        print("Consider manually compiling torch-radon")
        sys.exit(1)
else:
    print(not_found)
    print("You need to have PyTorch installed")
    sys.exit(1)

# CUDA
cuda_version = torch.version.cuda
print("CUDA:", cuda_version, end=" ")

if cuda_version in ["10.1", "10.2"]:
    print(ok)
else:
    print(error)
    print("Precompiled packages are build for CUDA 10.1 and 10.2")
    print("Consider manually compiling torch-radon")
    sys.exit(1)

python_version = f"cp{python_version}-cp{python_version}m"
package_url = f"{base_url}/cuda-{cuda_version}/torch-{torch_version}/torch_radon-{version}-{python_version}-linux_x86_64.whl"
install_command = f"pip install {package_url}"
print(f"{blue}Executing: {install_command}{reset}")
os.system(install_command)
