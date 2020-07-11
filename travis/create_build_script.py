script = [
    "cd /code",
    'eval "$(conda shell.bash hook)"',
    "mkdir output",
    "export CXX=g++"
]

for python in ["37", "36"]:
    for cuda in ["10.2", "10.1"]:
        for torch in ["1.5", "1.4"]:
            script += [
                "",
                f"# Python {python}, PyTorch {torch}, CUDA {cuda}",
                f"mkdir -p output/cuda-{cuda}/torch-{torch}",
                f"conda install -n py{python}cu{cuda.replace('.', '')} pytorch={torch} cudatoolkit={cuda} -c pytorch",
                f"source activate py{python}cu{cuda.replace('.', '')}",
                "python --version",
                f"python build.py clean",
                f"CUDA_HOME=/usr/local/cuda-{cuda} python setup.py bdist_wheel",
                f"mv dist/*.whl output/cuda-{cuda}/torch-{torch}/"
            ]

with open("/code/travis/do_build.sh", "w") as f:
    f.write("\n".join(script))
