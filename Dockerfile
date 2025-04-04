FROM jupyter/datascience-notebook:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /home/jovyan/city2graph
WORKDIR /home/jovyan/city2graph

EXPOSE 8888
