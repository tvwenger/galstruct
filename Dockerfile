FROM continuumio/miniconda3:latest

COPY environment.yaml /environment.yaml
RUN conda env create -f /environment.yaml
ENV PATH /opt/conda/envs/galstruct/bin:$PATH
ENV CONDA_DEFAULT_ENV galstruct
RUN echo "conda activate galstruct" >> ~/.bashrc
RUN pip install git+https://github.com/tvwenger/sbi.git@circular_imports
RUN pip install git+https://github.com/tvwenger/galstruct.git@master