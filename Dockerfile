FROM continuumio/miniconda3:latest

RUN apt-get update
RUN apt-get -y install build-essential pkg-config libopenblas-dev

COPY . /galstruct

RUN conda env create -f /galstruct/environment.yaml
ENV PATH /opt/conda/envs/galstruct/bin:$PATH
ENV CONDA_DEFAULT_ENV galstruct
RUN echo "conda activate galstruct" >> ~/.bashrc
RUN pip install --no-cache-dir git+https://github.com/tvwenger/sbi.git@circular_imports
RUN pip install --no-cache-dir /galstruct/.