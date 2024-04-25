# Dockerfile
# server image
# FROM nvidia/cuda:12.3.0-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Dockerfile Metadata
LABEL auther="Minyong Yoon <ycivil93@hanyang.ac.kr>"
LABEL version="0.0.3"
LABEL description="Profiling for huggingface-based Transformer and Stable Diffusion 1,2,XL"

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH="$PATH:/opt/conda/bin"

# update package
RUN apt-get update -y

# Install package dependencies
RUN apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcb-cursor0 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libxkbfile1 \
        openssh-client \
        wget \
        xcb \
        xkb-data

# nsight package install
RUN apt-get install -y --no-install-recommends \
        qtcreator \
        qtbase5-dev \
        qt5-qmake \
        cmake \
        cuda-nsight-compute-12-4 \
        cuda-nsight-systems-12-4 && \
    apt-get clean

# misc tool install
RUN apt-get install -y --no-install-recommends \
        nano \
        vim \
        tmux \
        git \
        ssh \
        openssh-server \
        libxrender-dev \
        bzip2 \
        libxext6 \
        libsm6 \
        mercurial \
        subversion

WORKDIR /home/workspace

COPY workspace/ /home/workspace

# conda install
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda create -n llm_profile && echo "conda activate llm_profile" >> ~/.bashrc

CMD [ "/bin/bash" ]
