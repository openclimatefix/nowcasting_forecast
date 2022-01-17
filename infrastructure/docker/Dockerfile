FROM python:3.9-slim

ARG TESTING=0

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
        libpq-dev \
    && apt-get autoclean && apt-get autoremove \
    > /dev/null

# copy files
COPY setup.py app/setup.py
COPY README.md app/README.md
COPY requirements.txt app/requirements.txt

# install requirements
RUN pip install -r app/requirements.txt

# copy library files
COPY nowcasting_forecast/ app/nowcasting_forecast/
COPY tests/ app/tests/

# change to app folder
WORKDIR /app

# install library
RUN pip install -e .

RUN if [ "$TESTING" = 1 ]; then pip install pytest; fi

CMD ["python", "-u","nowcasting_forecast/app.py"]
