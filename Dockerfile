# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry and add it to PATH
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry poetry

# Copy the rest of the application
COPY . .

RUN poetry lock
RUN poetry install --no-interaction --no-ansi

# Expose ports for FastAPI and Streamlit
EXPOSE 8500
EXPOSE 8501

RUN chmod +x start.sh
CMD ["./start.sh"]