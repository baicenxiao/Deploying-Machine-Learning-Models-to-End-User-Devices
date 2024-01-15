# Start from an Ubuntu base image
FROM ubuntu:20.04

# Set the maintainer label
LABEL maintainer="your-email@example.com"

# Avoid any prompts when installing packages
ENV DEBIAN_FRONTEND=noninteractive 

# Update and install essential packages
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    ca-certificates \
    python3-pip \
    python3-numpy \
    git \                  
    build-essential \      
    cmake \                
    curl \                 
    unzip \                
    xz-utils \             
    zip \
    pkg-config

# Download and extract ONNX Runtime v1.16.1 pre-built binaries
WORKDIR /onnxruntime
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-1.16.1.tgz && \
    tar xvzf onnxruntime-linux-x64-1.16.1.tgz && \
    rm onnxruntime-linux-x64-1.16.1.tgz

#Symlink the Library
RUN ln -s /onnxruntime/onnxruntime-linux-x64-1.16.1/lib/libonnxruntime.so.1.16.1 /usr/lib/

# Create a directory to mount your ONNX model from the host system
WORKDIR /model

# This command will run when the container starts
CMD ["/bin/bash"]
