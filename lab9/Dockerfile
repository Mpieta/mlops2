# Use a minimal Python image as the base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    curl libsnappy-dev make gcc g++ libc6-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv
RUN uv sync --locked

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Run the app with uv
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
