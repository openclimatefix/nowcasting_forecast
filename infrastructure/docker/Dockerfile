#FROM python:3.10-slim
#
#ARG TESTING=0
#
## make sure it doesnt fail if the docker file doesnt know the git commit
#ARG GIT_PYTHON_REFRESH=quiet
#
#RUN apt-get update
#RUN apt-get install git -y
#RUN apt-get install libgeos++-dev libproj-dev proj-data proj-bin -y
#
## copy files
#COPY setup.py app/setup.py
#COPY README.md app/README.md
#COPY requirements.txt app/requirements.txt
#RUN pip install git+https://github.com/SheffieldSolar/PV_Live-API#pvlive_api
#
#
## install requirements
#RUN pip install -r app/requirements.txt
#
## copy library files
#COPY nowcasting_forecast/ app/nowcasting_forecast/
#COPY tests/ app/tests/
#
## change to app folder
#WORKDIR /app
#
## install library
#RUN pip install -e .
#
#RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi
#
#CMD ["python", "-u","nowcasting_forecast/app.py"]
#
## Build: sudo docker build -t <project_name> .
## Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM ubuntu:latest

# Basic setup
RUN apt update && apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   libaio-dev \
                   libgeos++-dev \
                   libproj-dev \
                   && rm -rf /var/lib/apt/lists


ENV CONDA_ENV_NAME=nowcasting_forecast
ENV PYTHON_VERSION=3.10

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} cartopy eccodes numpy -c conda-forge


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Install requirements
COPY requirements.txt ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# copy library files
COPY nowcasting_forecast/ app/nowcasting_forecast/
COPY tests/ app/tests/

# change to app folder
WORKDIR /app

# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# install testing requirements
RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

# Cp in the development directory and install
COPY . ./
RUN source activate ${CONDA_ENV_NAME} && pip install -e .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "nowcasting_forecast", "/bin/bash", "-c"]

# Example commnad that can be used, need to set API_KEY, API_SECRET and SAVE_DIR
CMD ["conda", "run", "--no-capture-output", "-n", "nowcasting_forecast", "python", "-u","nowcasting_forecast/app.py"]
