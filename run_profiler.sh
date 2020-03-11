#!/usr/bin/env bash
sudo /usr/local/cuda/bin/nv-nsight-cu-cli -o $1 /home/ubuntu/miniconda3/bin/python benchmark.py
