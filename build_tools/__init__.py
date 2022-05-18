import sys
from glob import glob
import os
import shutil
import argparse
import subprocess
import re

from .generate_source import generate_source


def mapper(src, dst):
    paths = glob(src)

    src_pre = src.index("*")
    src_post = src_pre - len(src) + 1
    pre = dst[:dst.index("*")]
    post = dst[dst.index("*") + 1:]

    return [(path, pre + path[src_pre:src_post] + post) for path in paths]


def run(command):
    print(f"\u001b[34m{command}\u001b[0m")
    if os.system(command) != 0:
        print("\u001b[31mERROR IN COMPILATION\u001b[0m")
        exit(-1)


def run_compilation(files, f):
    for src, dst in files:
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            print(f"\u001b[32mCompiling {src}\u001b[0m")
            command = f(src, dst)
            run(command)
        else:
            print(f"\u001b[32mSkipping {src}\u001b[0m")


def render_template(src, dst):
    template_path = src
    cu_src_path = src[:-8] + "cu"

    # render template and generate CUDA source code
    generate_source(template_path, cu_src_path)

    return f"-c {cu_src_path} -o {dst}"


CXX_ADDITIONAL_FLAGS = []


def get_cuda_version(cuda_home):
    nvcc_out = subprocess.run([f"{cuda_home}/bin/nvcc", "--version"], stdout=subprocess.PIPE).stdout.decode('utf-8')
    m = re.search(r"V[0-9]+.[0-9]+", nvcc_out)
    str_version = m.group(0)[1:]

    return int(str_version.replace(".", ""))


def build(compute_capabilites=(60, 70, 75, 80, 86), verbose=False, cuda_home="/usr/local/cuda", cxx="g++"):
    cuda_version = get_cuda_version(cuda_home)
    nvcc = f"{cuda_home}/bin/nvcc"
    include_dirs = ["./include"]

    # compute capabilities >= 80 are only for cuda >= 11
    if cuda_version <= 110:
        compute_capabilities = [x for x in compute_capabilities if x < 80]

    cu_template_files = mapper("src/*.template", "objs/cuda/*.o")
    cu_files = mapper("src/*.cu", "objs/cuda/*.o")
    cpp_files = mapper("src/*.cpp", "objs/*.o")
    cpp_files = [x for x in cpp_files if x[0] != "src/pytorch.cpp"]

    all_objects = [y for x, y in cu_files + cpp_files]

    include_flags = [f"-I{x}" for x in include_dirs]
    cxx_flags = ["-std=c++11 -fPIC -static -static-libgcc -static-libstdc++"] + include_flags + ["-O3"]
    nvcc_flags = ["-std=c++11", f"-ccbin={cxx}", "-Xcompiler", "-fPIC", "-Xcompiler -static",
                  "-Xcompiler -static-libgcc", "-Xcompiler -static-libstdc++"] + include_flags + \
        [f"-gencode arch=compute_{x},code=sm_{x}" for x in compute_capabilites] + [
        "-DNDEBUG -O3 --generate-line-info --compiler-options -Wall"]

    if verbose:
        cxx_flags.append("-DVERBOSE")
        nvcc_flags.append("-DVERBOSE")

    cxx_flags = " ".join(cxx_flags)
    nvcc_flags = " ".join(nvcc_flags)

    # create output directory
    if not os.path.exists("objs/cuda"):
        os.makedirs("objs/cuda")

    # compile
    run_compilation(cu_template_files, lambda src, dst: f"{nvcc} {nvcc_flags} {render_template(src, dst)}")
    run_compilation(cu_files, lambda src, dst: f"{nvcc} {nvcc_flags} -c {src} -o {dst}")
    run_compilation(cpp_files, lambda src, dst: f"{cxx} {cxx_flags} -c {src} -o {dst}")

    run(f"ar rc objs/libradon.a {' '.join(all_objects)}")


def clean():
    print(f"\u001b[32mCleaning\u001b[0m")
    if os.path.exists("objs"):
        shutil.rmtree("objs")


if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument("-cuda-home", default=os.getenv("CUDA_HOME", "/usr/local/cuda"))
    #
    # args = args.parse_args()
    # print(args.cuda_home)

    if len(sys.argv) < 2 or sys.argv[1] == "build":
        build(cuda_home=os.getenv("CUDA_HOME", "/usr/local/cuda"))
    elif sys.argv[1] == "clean":
        clean()
