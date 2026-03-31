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


def _parse_arch_token(token):
    token = token.strip()
    if not token:
        return None, False

    use_ptx = token.endswith("+PTX")
    token = token.replace("+PTX", "")
    token = token.replace("sm_", "").replace("compute_", "")

    if "." in token:
        # e.g. 8.9 -> 89, 12.0 -> 120
        parts = token.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            arch = int(parts[0]) * 10 + int(parts[1])
            return arch, use_ptx

    if token.isdigit():
        return int(token), use_ptx

    return None, False


def _parse_arch_list(arch_list):
    if not arch_list:
        return [], set()

    tokens = re.split(r"[;,\s]+", arch_list)
    sms = []
    ptx = set()
    for token in tokens:
        arch, use_ptx = _parse_arch_token(token)
        if arch is None:
            continue
        sms.append(arch)
        if use_ptx:
            ptx.add(arch)
    return sms, ptx


def _get_nvcc_supported_sms(cuda_home):
    nvcc = f"{cuda_home}/bin/nvcc"
    outputs = []

    for args in (["--list-gpu-code"], ["--help"]):
        try:
            out = subprocess.run([nvcc] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            outputs.append(out.stdout.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    supported = set()
    for out in outputs:
        for m in re.findall(r"sm_([0-9]+)", out):
            supported.add(int(m))

    return sorted(supported)


def _resolve_compute_capabilities(user_compute_capabilities, cuda_home):
    default_sms = [60, 70, 75, 80, 86, 89, 90, 100, 120]
    requested_ptx = set()

    # Priority: TORCH_CUDA_ARCH_LIST, then CUDAARCHS, then function parameter/defaults.
    env_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST") or os.getenv("CUDAARCHS")
    if env_arch_list:
        requested_sms, requested_ptx = _parse_arch_list(env_arch_list)
    elif user_compute_capabilities is not None:
        requested_sms = list(user_compute_capabilities)
    else:
        requested_sms = default_sms

    supported_sms = _get_nvcc_supported_sms(cuda_home)

    # If nvcc can tell us its supported arches, keep only those.
    if supported_sms:
        requested_sms = [sm for sm in requested_sms if sm in supported_sms]
        requested_ptx = {sm for sm in requested_ptx if sm in supported_sms}

    # Add currently visible GPU capability when available.
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            device_sm = major * 10 + minor
            if (not supported_sms) or (device_sm in supported_sms):
                requested_sms.append(device_sm)
    except Exception:
        pass

    requested_sms = sorted(set(requested_sms))

    # If no explicit +PTX was requested, add PTX for the highest arch as a fallback.
    if requested_sms and not requested_ptx:
        requested_ptx = {requested_sms[-1]}

    if not requested_sms and supported_sms:
        requested_sms = [supported_sms[-1]]
        requested_ptx = {supported_sms[-1]}

    return requested_sms, requested_ptx, supported_sms


def build(compute_capabilities=None, verbose=False, cuda_home="/usr/local/cuda", cxx="g++"):
    cuda_version = get_cuda_version(cuda_home)
    nvcc = f"{cuda_home}/bin/nvcc"
    include_dirs = ["./include"]

    compute_capabilities, ptx_capabilities, supported_sms = _resolve_compute_capabilities(
        compute_capabilities,
        cuda_home
    )

    if len(compute_capabilities) == 0:
        print("\u001b[31mCould not determine CUDA architectures for nvcc.\u001b[0m")
        exit(-1)

    if supported_sms:
        print(f"\u001b[32mnvcc supported SMs: {supported_sms}\u001b[0m")
    print(f"\u001b[32mBuilding for SMs: {compute_capabilities}, PTX: {sorted(ptx_capabilities)}\u001b[0m")

    cu_template_files = mapper("src/*.template", "objs/cuda/*.o")
    cu_files = mapper("src/*.cu", "objs/cuda/*.o")
    cpp_files = mapper("src/*.cpp", "objs/*.o")
    cpp_files = [x for x in cpp_files if x[0] != "src/pytorch.cpp"]

    all_objects = [y for x, y in cu_files + cpp_files]

    include_flags = [f"-I{x}" for x in include_dirs]
    cxx_flags = ["-std=c++11 -fPIC -static -static-libgcc -static-libstdc++"] + include_flags + ["-O3"]
    gencode_flags = [f"-gencode arch=compute_{x},code=sm_{x}" for x in compute_capabilities]
    gencode_flags += [f"-gencode arch=compute_{x},code=compute_{x}" for x in sorted(ptx_capabilities)]
    nvcc_flags = ["-std=c++11", f"-ccbin={cxx}", "-Xcompiler", "-fPIC", "-Xcompiler -static",
                  "-Xcompiler -static-libgcc", "-Xcompiler -static-libstdc++"] + include_flags + \
        gencode_flags + [
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
