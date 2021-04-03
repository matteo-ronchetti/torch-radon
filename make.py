import sys
from glob import glob
import os
import shutil
import json

from ptx_annotation import annotate_ptx


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


def build(compute_capabilites=(60, 70, 75, 80, 86), debug=False, cuda_home="/usr/local/cuda", cxx="g++",
          keep_intermediate=True):

    cuda_major_version = int(json.load(open(os.path.join(cuda_home, "version.json")))["cuda"]["version"].split(".")[0])
    nvcc = f"{cuda_home}/bin/nvcc"
    include_dirs = ["./include"]
    intermediate_dir = "intermediates"

    # compute capabilities >= 80 are only for cuda >= 11
    if cuda_major_version < 11:
        compute_capabilites = [x for x in compute_capabilites if x < 80]

    cu_files = mapper("src/*.cu", "objs/cuda/*.o")
    cpp_files = mapper("src/*.cpp", "objs/*.o")
    cpp_files = [x for x in cpp_files if x[0] != "src/pytorch.cpp"]

    all_objects = [y for x, y in cu_files + cpp_files]

    opt_flags = ["-g", "-DVERBOSE"] if debug else ["-DNDEBUG", "-O3"]

    include_flags = [f"-I{x}" for x in include_dirs]
    cxx_flags = ["-std=c++11 -fPIC -static"] + include_flags + opt_flags
    nvcc_base_flags = ["-std=c++11", f"-ccbin={cxx}", "-Xcompiler", "-fPIC", "-Xcompiler -static",
                       "-Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0"] + include_flags + opt_flags + [
                           "--generate-line-info --compiler-options -Wall --use_fast_math"]
    nvcc_flags = nvcc_base_flags + [f"-gencode arch=compute_{x},code=sm_{x}" for x in compute_capabilites]

    if keep_intermediate:
        if not os.path.exists(intermediate_dir):
            os.mkdir(intermediate_dir)
        nvcc_flags.append(f"-keep --keep-dir {intermediate_dir}")

    cxx_flags = " ".join(cxx_flags)
    nvcc_flags = " ".join(nvcc_flags)

    # create output directory
    if not os.path.exists("objs/cuda"):
        os.makedirs("objs/cuda")

    # compile
    run_compilation(cu_files, lambda src, dst: f"{nvcc} {nvcc_flags} -c {src} -o {dst}")
    run_compilation(cpp_files, lambda src, dst: f"{cxx} {cxx_flags} -c {src} -o {dst}")

    run(f"ar rc objs/libradon.a {' '.join(all_objects)}")

    if keep_intermediate:
        for path in os.listdir(intermediate_dir):
            if path.endswith(".ptx"):
                annotate_ptx(os.path.join(intermediate_dir, path))
            else:
                os.remove(os.path.join(intermediate_dir, path))


def clean():
    print(f"\u001b[32mCleaning\u001b[0m")
    for f in ["objs", "build", "dist", "intermediates", "torch_radon.egg-info"]:
        if os.path.exists(f):
            print(f"Removing '{f}'")
            shutil.rmtree(f)


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
