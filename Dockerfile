FROM continuumio/miniconda3:latest

WORKDIR /galstruct
COPY . /galstruct
RUN conda env create -f /galstruct/environment.yaml
ENV PATH /opt/conda/envs/galstruct/bin:$PATH
ENV CONDA_DEFAULT_ENV galstruct
RUN echo "conda activate galstruct" >> ~/.bashrc
RUN pip install /galstruct/.
