FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install pytorch cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all
