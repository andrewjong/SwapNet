FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update 

RUN apt-get install -y \
    build-essential wget

RUN apt-get install -y git
RUN apt-get install -y curl

WORKDIR /app/

RUN echo "Installing and creating Miniconda environment..."
# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& chmod +x /miniconda.sh \
	&& /miniconda.sh -b -p /miniconda \
	&& rm /miniconda.sh \
	&& echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH=/miniconda/bin:$PATH

RUN mkdir SwapNet 

# Create the environment, set to activate automatically
RUN cd SwapNet && \
	wget https://raw.githubusercontent.com/andrewjong/SwapNet/master/environment.yml \
	&& conda env create \
	&& echo "source activate swapnet" >> ~/.bashrc

# Checking environment, required for ROI to build properly
# this command should display gpu properties
RUN /bin/bash -c "nvidia-smi || echo 'nvidia-smi failed. A GPU is necessary to properly compile the ROI dependency. Make sure you have the NVIDIA Container Toolkit installed and enabled-by-default by editing /etc/docker/daemon.json'"

# This should print true
RUN /bin/bash -c "source activate swapnet && python -c 'import torch; print(torch.cuda.is_available())'"

# CUDA Home should not be none
RUN /bin/bash -c "source activate swapnet && python -c 'import torch;from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)'"

# ROI Dependency
RUN echo "Compiling ROI dependency..."

RUN git clone https://github.com/jwyang/faster-rcnn.pytorch.git # clone to a SEPARATE project directory

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch && git checkout pytorch-1.0 && pip install -r requirements.txt"

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch/lib/pycocotools && wget https://raw.githubusercontent.com/muaz-urwa/temp_files/master/setup.py && python setup.py build_ext --inplace"

RUN  /bin/bash -c "source activate swapnet && cd faster-rcnn.pytorch/lib && python setup.py build develop"

RUN  /bin/bash -c "source activate swapnet && ln -s /app/faster-rcnn.pytorch/lib /app/SwapNet/lib"

RUN  /bin/bash -c "source activate swapnet && conda install seaborn"

RUN echo "Done!"
