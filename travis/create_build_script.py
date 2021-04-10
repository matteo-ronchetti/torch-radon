import re
import urllib.request
import urllib.parse


url = 'https://download.pytorch.org/whl/torch_stable.html'
f = urllib.request.urlopen(url)
txt = f.read().decode('utf-8')

regex = r'<a\s*href\s*=\s*"[^"]*"\s*>cu([0-9]+)\/torch-([0-9.]+)[^-]*-cp([0-9]{2})'

configs = []
for cu, pt, py in re.findall(regex, txt):
    cuda = int(cu)
    python = int(py)
    pytorch = int(pt[:3].replace(".", ""))
    
    if cuda >= 100 and python >= 36 and pytorch >= 15:
        configs.append((cuda, python, pytorch))
        
configs = list(set(configs))
python_and_cuda = list(set([(cuda, python) for cuda, python, _ in configs]))

# print(f"Need to create {len(python_and_cuda)} environments")
print(f"Need to compile {len(configs)} packages")

script = [
    "cd /code",
    'eval "$(/root/miniconda3/bin/conda shell.bash hook)"',
    "mkdir output",
    "export CXX=g++"
]

# for cuda, python in python_and_cuda:
#     cuda_full = f"{cuda // 10}.{cuda % 10}"
#     python_full = f"{python // 10}.{python % 10}"
#     script += [
#         "",
#         f"# Virtualenv Python {python}, CUDA {cuda_full}",
#         f"conda create -n py{python}cu{cuda} python={python_full}"
#     ]

for cuda, python, torch in configs:
    cuda_full = f"{cuda // 10}.{cuda % 10}"
    python_full = f"{python // 10}.{python % 10}"
    torch_full = f"{torch // 10}.{torch % 10}"
    env = f"py{python}cu{cuda}pt{torch}"

    script += [
        "",
        f"# Python {python}, PyTorch {torch_full}, CUDA {cuda_full}",
        f"mkdir -p output/cuda-{cuda_full}/torch-{torch_full}",
        f"conda create -n {env} python={python_full}",
        f"conda install -n {env} pytorch={torch_full} cudatoolkit={cuda_full} -c pytorch -c conda-forge",
        f"source /root/miniconda3/bin/activate {env}",
        "pip install Demangler",
        "python --version",
        'python -c "import torch; print(torch.version.cuda)"',
        "python make.py clean",
        f"export CUDA_HOME=/usr/local/cuda-{cuda_full}",
        # force recompilation otherwise will reuse builds even if CUDA version changes
        "python setup.py build_ext --force",
        "python setup.py bdist_wheel",
        f"mv dist/*.whl output/cuda-{cuda_full}/torch-{torch_full}/"
    ]

with open("/code/travis/do_build.sh", "w") as f:
    f.write("\n".join(script))
