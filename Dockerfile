FROM jupyter/datascience-notebook:latest

USER root

# Install uv, the new fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /home/jovyan/city2graph

# Install dependencies using uv sync, leveraging Docker layer caching.
# This installs dependencies from pyproject.toml into a separate layer
# before the application code is copied, improving build times.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --system --extra cpu --group dev --no-install-project

# Copy the rest of the application code
COPY . .

# Install the project in editable mode. Dependencies are already installed, so this is fast.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --system --extra cpu --group dev

# Ensure the jovyan user owns the files
RUN chown -R jovyan:users /home/jovyan/city2graph

# Switch back to the non-root user
USER jovyan

EXPOSE 8888
