# Use a specific Python image
FROM python:3.10.13-slim

# Set Poetry environment
RUN pip install poetry \
    && poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy only the necessary files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install

# Copy the entire source code
COPY ./app /app

# Expose the necessary port
EXPOSE 8080

# Entrypoint for your application
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

