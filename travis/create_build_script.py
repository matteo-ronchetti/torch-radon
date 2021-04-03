script = [
    "cd /code",
    'eval "$(/opt/conda/bin/conda shell.bash hook)"',
    "mkdir output",
    "export CXX=g++"
]

for python in ["38", "37"]:  # , "36"
    for cuda in ["11.0"]:  # , "10.2", "10.1"
        for torch in ["1.7"]:
            script += [
                "",
                f"# Python {python}, PyTorch {torch}, CUDA {cuda}",
                f"mkdir -p output/cuda-{cuda}/torch-{torch}",
                f"conda install -n py{python}cu{cuda.replace('.', '')} pytorch={torch} cudatoolkit={cuda} -c pytorch",
                f"source /opt/conda/bin/activate py{python}cu{cuda.replace('.', '')}",
                "python --version",
                'python -c "import torch; print(torch.version.cuda)"',
                "python build.py clean",
                f"export CUDA_HOME=/usr/local/cuda-{cuda}",
                # force recompilation otherwise will reuse builds even if CUDA version changes
                "python setup.py build_ext --force",
                "python setup.py bdist_wheel",
                f"mv dist/*.whl output/cuda-{cuda}/torch-{torch}/"
            ]

with open("/code/travis/do_build.sh", "w") as f:
    f.write("\n".join(script))
