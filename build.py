import sys
from glob import glob
import os
import shutil
import argparse

from build_tools import build



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
