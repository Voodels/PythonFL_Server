# Federated Learning with Flower - AI Coding Instructions

## Architecture Overview

This is a **Flower-based federated learning system** with MNIST classification, featuring:
- **1 Server** (`fl_server/`) - Orchestrates FL rounds using custom `RichStrategy` 
- **2+ Clients** (`fl_client/`) - Train local models and share updates
- **Docker Compose orchestration** - Single command deployment with networking

### Key Components
- **Shared Model**: Simple Dense NN for MNIST (28x28 → 128 → 10) in both `*/utils.py`
- **Rich UI**: Server shows live table + spinners; clients show status during training
- **Transfer Metrics**: Clients report `up_bytes`/`down_bytes` for model weight transfers

## Critical Development Patterns

### Docker-First Development
```bash
# Primary workflow - builds and runs entire FL system
docker-compose build --no-cache  # After code changes
docker-compose up -d             # Detached mode
docker-compose logs -f fl-server # Monitor server logs
```

**Port Configuration**: Server exposed on `8081:8080` (host:container) to avoid Adminer conflicts.

### Dependency Management
- **tensorflow-cpu==2.13.0** (not full TF) - reduces build time/size
- **numpy==1.24.3** (pinned) - avoids compatibility issues
- **Rich UI libs** synchronized across server/client
- **Dockerfile pip hardening**: `PIP_DEFAULT_TIMEOUT=1000` + `PIP_RETRIES=10`

### Flower Framework Integration
- Server uses **custom `RichStrategy(FedAvg)`** - overrides `aggregate_fit/evaluate` for UI logging
- Clients extend **`NumPyClient`** - implements `fit()`, `evaluate()`, `get_parameters()`
- **Environment-based networking**: `FLOWER_SERVER_HOST=fl-server` (Docker service name)
- **Client differentiation**: `CLIENT_ID` env var for logging/debugging

## UI/Logging Conventions

### Rich Console Integration
- **Server**: `ServerUI` context manager with `Live` table + transient spinners
- **Clients**: `console.status()` spinners during training/evaluation phases
- **Dual logging**: Rich UI + standard logs (visible in `docker logs`)

### Transfer Metrics Pattern
```python
# Clients compute and return transfer sizes
metrics: Dict[str, Scalar] = {"up_bytes": int(up_bytes), "down_bytes": int(down_bytes)}
return new_weights, len(x_train), metrics

# Server aggregates and displays transfer summaries  
total_up += int(fitres.metrics.get("up_bytes", 0))
self.ui.log(f"Round {round}", f"↑ updates ~{up_mb:.2f} MB")
```

## Common Issues & Solutions

### Rich Live Display Conflicts
- **Never run multiple `Live()` contexts simultaneously**
- Use `console.status()` for transient spinners instead of `Progress`
- Server UI must be singleton context manager

### TensorFlow in Docker
- Import keras via `getattr(tf, "keras")` to avoid Pylance typing issues
- Data preprocessing: `x_train / 255.0` normalization is critical
- Model compilation must happen in both server/client `utils.py`

### Container Networking
- Clients connect to `fl-server:8080` (service name, internal port)
- Host accesses server on `localhost:8081` (mapped port)
- Use `depends_on` to ensure server starts first

## Testing & Debugging

### Validation Workflow
```bash
# Check container status
docker-compose ps

# View aggregated logs with transfer metrics
docker-compose logs --tail=50 fl-server

# Monitor specific client training
docker-compose logs -f fl-client-1
```

### Expected Log Patterns
- Server: "Round X | Aggregating results from 2 clients..." → "↑ updates ~X.XX MB"
- Clients: "Training locally (1 epoch)..." → "Accuracy: 0.XXXX"
- **3 rounds total** - system shuts down gracefully after completion

## File Structure Patterns
- **Symmetric `requirements.txt`** - server/client have identical dependencies
- **Duplicated `utils.py`** - Docker build contexts require separate copies
- **Environment-driven config** - no hardcoded IPs/ports in code
- **Rich markup in logs** - use `[bold cyan]`, `[red]`, etc. for colored output
