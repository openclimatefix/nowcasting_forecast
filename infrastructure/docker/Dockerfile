FROM python:3.9-slim as build

ARG TESTING=0

# make sure it doesnt fail if the docker file doesnt know the git commit
ARG GIT_PYTHON_REFRESH=quiet

WORKDIR /app
ENV PATH="/root/.local/bin:/opt/venv/:$PATH" VIRTUAL_ENV="/opt/venv"

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    curl \
    git  \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN poetry config virtualenvs.create false

RUN python3 -m venv /opt/venv

COPY pyproject.toml poetry.lock README.md /app/
COPY nowcasting_forecast/ /app/nowcasting_forecast/

RUN poetry install

RUN if [ "$TESTING" -eq 1 ]; then\
    poetry install --extras test;\
  fi

FROM python:3.9-slim as runtime

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV="/opt/venv"

COPY --from=build /opt/venv /opt/venv
COPY nowcasting_forecast/ /app/nowcasting_forecast/
COPY tests/ /app/tests/

ENV PYTHONPATH="/app/nowcasting_forecast"

CMD ["python3","-u", "nowcasting_forecast/app.py"]