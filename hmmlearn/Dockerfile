FROM continuumio/miniconda3

MAINTAINER s0r4syl <ckb15002@nitech.jp>

# docker build
RUN echo "now building..."
RUN apt update
RUN apt upgrade -y
RUN apt install -y vim build-essential
RUN conda update -n base conda
RUN pip install numpy==1.16.2
RUN pip install -U --user hmmlearn
