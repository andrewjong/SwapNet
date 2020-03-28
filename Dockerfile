FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update 

RUN apt-get install -y \
    build-essential wget

RUN apt-get install -y git
RUN apt-get install -y curl

WORKDIR /app/

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

RUN git clone https://github.com/andrewjong/SwapNet.git

ENV PATH=/miniconda/bin:$PATH

RUN cd SwapNet && conda env create 

RUN git clone https://github.com/jwyang/faster-rcnn.pytorch.git # clone to a SEPARATE project directory

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch && git checkout pytorch-1.0 && pip install -r requirements.txt"

RUN echo "Compiling roi dependency"

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch/lib/pycocotools && wget https://raw.githubusercontent.com/muaz-urwa/temp_files/master/setup.py && python setup.py build_ext --inplace"

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch/lib && python setup.py build develop"

RUN  /bin/bash -c "source activate swapnet && ln -s /app/faster-rcnn.pytorch/lib /app/SwapNet/lib"

RUN  /bin/bash -c "source activate swapnet && conda install seaborn"

# checking environment
# this command should display gpu properties
RUN nvidia-smi

# This should print true
RUN /bin/bash -c "source activate swapnet && python -c 'import torch; print(torch.cuda.is_available())'"

# CUDA Home should not be none
RUN /bin/bash -c "source activate swapnet && python -c 'import torch;from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)'"

