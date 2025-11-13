# Single-stage build for Indian Stock Analysis MCP Server
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcp

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY --chown=mcp:mcp server.py stock_analyzer.py ./

# Switch to non-root user
USER mcp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV LOG_LEVEL=ERROR


# Health check to verify container is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - use uv run to ensure dependencies are available
CMD ["uv", "run", "python", "server.py"]