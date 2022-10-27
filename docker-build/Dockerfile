FROM ubuntu:18.04
FROM continuumio/miniconda3
#FROM nvidia/cuda:11.6.1-base-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update

# ---
# preload_model.py : error - from realesrgan import RealESRGANer 
# error message - ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# ---
RUN apt-get install libgl1-mesa-glx -y

WORKDIR /
COPY . /InvokeAI/

WORKDIR /InvokeAI

RUN mkdir -p models/ldm/stable-diffusion-v1

RUN mv model.ckpt models/ldm/stable-diffusion-v1/model.ckpt

RUN conda env create -f environment.yml
RUN echo "succeed = environment.yml"
SHELL ["conda", "run", "-n", "invokeai", "/bin/bash", "-c"]
RUN echo "succeed = activate invokeai"
RUN python3 scripts/preload_models.py
RUN echo "succeed = preload_models.py"

EXPOSE 7860

# RUN dos2unix docker-build/entrypoint.sh
RUN chmod +x docker-build/entrypoint.sh

ENTRYPOINT docker-build/entrypoint.sh
