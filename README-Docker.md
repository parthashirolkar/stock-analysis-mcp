# Docker Setup for Indian Stock Analysis MCP Server

This guide explains how to run the Indian Stock Analysis MCP server using Docker Desktop on Windows, eliminating the need for WSL startup.

## Prerequisites

- Docker Desktop installed and running on Windows
- Claude Desktop for MCP integration
- Git (for cloning the repository)

## Quick Start

### 1. Build and Run the Container

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd stock-analysis-mcp

# Switch to the Docker branch
git checkout feature/docker-containerization

# Build and run using Docker Compose
docker-compose up -d
```

### 2. Configure Claude Desktop

Add the following to your Claude Desktop MCP configuration file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "indian-stock-analysis": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "indian-stock-mcp-server",
        "python",
        "server.py"
      ]
    }
  }
}
```

### 3. Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

## Container Management

### Start the Container
```bash
docker-compose up -d
```

### Stop the Container
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f stock-analysis-mcp
```

### Monitor Container Status
```bash
docker-compose ps
```

### Check Health Status
```bash
docker inspect indian-stock-mcp-server | grep -A 10 "Health"
```

## Docker Configuration Details

### Auto-start Configuration
The container is configured with `restart: unless-stopped`, which means:
- Container starts automatically when Docker Desktop starts
- Container restarts automatically if it crashes
- Container only stops when manually stopped

### Resource Limits
- **Memory Limit**: 512MB
- **Memory Reservation**: 256MB
- **Health Check**: Every 30 seconds

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensures immediate log output
- `LOG_LEVEL=ERROR`: Sets logging to error level only
- `TZ=Asia/Kolkata`: Sets timezone to Indian Standard Time

## Building from Scratch

If you need to rebuild the container:

```bash
# Build the image
docker-compose build

# Or build without cache
docker-compose build --no-cache
```

## Troubleshooting

### Container Won't Start
```bash
# Check container logs
docker-compose logs stock-analysis-mcp

# Check if container is running
docker ps | grep indian-stock-mcp
```

### MCP Server Not Responding
1. Verify the container is running: `docker-compose ps`
2. Check logs for errors: `docker-compose logs stock-analysis-mcp`
3. Test the MCP server directly:
   ```bash
   docker exec -i indian-stock-mcp-server python server.py
   ```

### Memory Issues
The container is limited to 512MB memory. If you experience issues, you can increase this limit in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 1G  # Increased from 512M
```

### Rebuilding After Changes
```bash
# Stop the container
docker-compose down

# Rebuild and start
docker-compose up -d --build
```

## Monitoring in Docker Desktop

You can monitor the container directly in Docker Desktop:

1. **Containers Tab**: View status, resource usage, and logs
2. **Health Status**: Green checkmark indicates healthy container
3. **Resource Usage**: Monitor CPU and memory consumption
4. **Logs**: View real-time logs from the container

## Updates and Maintenance

### Update the MCP Server
1. Pull the latest changes: `git pull origin feature/docker-containerization`
2. Rebuild and restart: `docker-compose up -d --build`

### Clean Up Old Images
```bash
# Remove unused images
docker image prune -f

# Remove all unused Docker resources
docker system prune -f
```

## Security Considerations

- The container runs as a non-root user (`mcp`)
- No ports are exposed (STDIO transport only)
- Minimal base image (Python 3.13-slim)
- No sensitive data is stored in the container

## Performance Notes

- First request may be slower (container cold start)
- Subsequent requests should be responsive
- Container uses minimal system resources
- Network requests are made to Yahoo Finance API

## Integration with Other Tools

Since this runs in Docker, you can easily integrate with:
- **Docker Compose** for multi-service setups
- **Kubernetes** for orchestration
- **Monitoring tools** like Prometheus/Grafana
- **Log aggregation** with ELK stack or similar

## Support

For issues related to:
- **Docker**: Check Docker Desktop documentation
- **MCP Server**: Review the main README.md
- **Indian Stock Data**: Verify internet connectivity and Yahoo Finance API status