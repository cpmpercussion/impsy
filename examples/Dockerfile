FROM python:3.11.5-slim-bookworm
# FROM tensorflow/tensorflow:2.15.0

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=1.8.3

# ensure curl
RUN apt-get update \
  && apt-get install --no-install-recommends -y curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# System deps:
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy only requirements to cache them in docker layer
WORKDIR /impsy
COPY ./poetry.lock ./pyproject.toml /impsy/

# # Run Poetry Install
RUN poetry install --no-interaction --no-ansi \
  && poetry cache clear --all .

# # Creating folders, and files for a project:
COPY . /impsy

CMD ["bash"]
