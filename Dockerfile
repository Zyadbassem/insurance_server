# Build stage: use dev image for package installation
FROM dhi.io/python:3.14-dev AS builder

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Runtime stage: minimal hardened image, no package manager
FROM dhi.io/python:3.14

WORKDIR /app

# Copy Python site-packages from builder
COPY --from=builder /usr/lib/python3.14/site-packages /usr/lib/python3.14/site-packages

# Copy application code and model
COPY main.py model.joblib ./

# Expose the port
EXPOSE 8000

# Start the FastAPI application
CMD ["python", "-m", "uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]
