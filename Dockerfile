FROM jupyter/datascience-notebook:latest

COPY environment.yml /tmp/environment.yml
RUN conda update -n base conda -c defaults -y &&\
    conda env create -f /tmp/environment.yml &&\
    conda clean -a

COPY . /home/jovyan/city2graph
WORKDIR /home/jovyan/city2graph

EXPOSE 8888
