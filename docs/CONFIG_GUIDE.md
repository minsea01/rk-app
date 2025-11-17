# Configuration Management Guide

## Overview

RK-App uses a **unified configuration system** with a clear **priority chain** to eliminate configuration conflicts and debugging headaches.

**Configuration Priority (highest to lowest):**
1. **CLI arguments** - Runtime overrides (`--imgsz 640`)
2. **Environment variables** - System-level config (`RK_IMGSZ=640`)
3. **YAML config file** - Project-level config (`config/app.yaml`)
4. **Python defaults** - Code defaults (`apps/config.py`)

**Key Principle:** Higher priority sources **always win**. CLI > ENV > YAML > Python.

---

## Quick Start

### Example 1: Use default configuration

```bash
# Uses Python defaults (imgsz=416, conf_threshold=0.5)
python scripts/run_rknn_sim.py
```

### Example 2: Override with CLI arguments

```bash
# CLI overrides defaults (imgsz=640)
python scripts/run_rknn_sim.py --imgsz 640 --conf 0.25
```

### Example 3: Override with environment variables

```bash
# ENV overrides YAML and defaults
export RK_IMGSZ=640
export RK_CONF_THRESHOLD=0.25
python scripts/run_rknn_sim.py
```

### Example 4: Use YAML configuration

```yaml
# config/app.yaml
imgsz: 640
conf_threshold: 0.25
```

```bash
# Uses values from config/app.yaml
python scripts/run_rknn_sim.py
```

### Example 5: Priority chain in action

```yaml
# config/app.yaml
imgsz: 320
```

```bash
# CLI (640) > ENV (416) > YAML (320) > Python (416)
# Result: imgsz=640
export RK_IMGSZ=416
python scripts/run_rknn_sim.py --imgsz 640
```

---

## Configuration Sources

### 1. CLI Arguments (Highest Priority)

Runtime overrides using argparse:

```bash
python scripts/run_rknn_sim.py \
  --model artifacts/models/yolo11n.onnx \
  --imgsz 640 \
  --conf 0.5 \
  --iou 0.45
```

**When to use:** One-off experiments, testing different parameters.

---

### 2. Environment Variables

System-level configuration using `RK_*` prefix:

```bash
# Set environment variables
export RK_IMGSZ=640
export RK_CONF_THRESHOLD=0.5
export RK_IOU_THRESHOLD=0.45
export RK_LOG_LEVEL=DEBUG

# Run script (uses ENV values)
python scripts/run_rknn_sim.py
```

**Naming convention:** `RK_` + `UPPERCASE_KEY`
- `imgsz` → `RK_IMGSZ`
- `conf_threshold` → `RK_CONF_THRESHOLD`
- `target_platform` → `RK_TARGET_PLATFORM`

**When to use:** Docker containers, CI/CD pipelines, board deployment.

---

### 3. YAML Configuration File

Project-level configuration in `config/app.yaml`:

```yaml
# config/app.yaml
log_level: INFO

# Model configuration
imgsz: 416
conf_threshold: 0.5
iou_threshold: 0.45

# RKNN configuration
target_platform: rk3588
optimization_level: 3
core_mask: 7

# Video configuration
camera_id: 0
fps: 30

# Network configuration
tcp_port: 8080
udp_port: 8081
```

**When to use:** Shared team settings, deployment environments.

---

### 4. Python Defaults (Lowest Priority)

Code-level defaults in `apps/config.py`:

```python
from apps.config import ModelConfig

# Access defaults
print(ModelConfig.DEFAULT_SIZE)  # 416
print(ModelConfig.CONF_THRESHOLD_DEFAULT)  # 0.5
```

**When to use:** Fallback values, recommended settings.

---

## Using ConfigLoader

### Basic Usage

```python
from apps.config_loader import ConfigLoader

# Create loader (reads config/app.yaml)
loader = ConfigLoader()

# Get value with priority chain
imgsz = loader.get('imgsz', cli_value=None, default=416)
# Checks: CLI → ENV (RK_IMGSZ) → YAML → default (416)
```

### Get Model Configuration

```python
from apps.config_loader import ConfigLoader

loader = ConfigLoader()

# Method 1: With CLI values
model_config = loader.get_model_config(
    imgsz=640,  # From argparse
    conf_threshold=0.5,
    iou_threshold=0.45,
)
# Returns: {'imgsz': 640, 'conf_threshold': 0.5, ...}

# Method 2: Auto-detect from ENV/YAML/defaults
model_config = loader.get_model_config()
# Checks priority chain for each parameter
```

### Get RKNN Configuration

```python
rknn_config = loader.get_rknn_config(
    target_platform='rk3588',  # Optional CLI override
    optimization_level=3,
)
# Returns: {'target_platform': 'rk3588', 'optimization_level': 3, ...}
```

### Type Validation

ConfigLoader automatically validates types and ranges:

```python
# Valid
imgsz = loader.get('imgsz', default=416, value_type=int)  # ✅ 416

# Invalid - raises ValidationError
imgsz = loader.get('imgsz', default='not_a_number', value_type=int)  # ❌
```

### Custom Validation

```python
def validate_imgsz(value):
    if value not in [416, 640]:
        raise ValueError(f"Image size must be 416 or 640, got {value}")

imgsz = loader.get(
    'imgsz',
    default=416,
    value_type=int,
    validate=validate_imgsz  # Custom validation
)
```

---

## Integration Example

### Before (Fragile)

```python
# Unclear priority, magic numbers, no validation
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640)  # Wait, is default 640 or 416?
args = parser.parse_args()

imgsz = args.imgsz  # What if ENV variable exists? Which wins?
```

### After (Production-Grade)

```python
import argparse
from apps.config_loader import ConfigLoader

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, help='Image size (default: from config)')
parser.add_argument('--conf', type=float, help='Confidence threshold')
args = parser.parse_args()

# Load configuration with priority chain
loader = ConfigLoader(config_file='config/app.yaml')
config = loader.get_model_config(
    imgsz=args.imgsz,  # CLI (highest priority)
    conf_threshold=args.conf,
)

# Log final configuration
logger.info(loader.dump_config(config))
# Output:
#   Configuration:
#     conf_threshold: 0.5 (source: YAML)
#     imgsz: 640 (source: CLI)
#     iou_threshold: 0.45 (source: DEFAULT)
#     max_detections: 8400
```

---

## Migration Guide

### Updating Existing Scripts

**Step 1:** Import ConfigLoader

```python
from apps.config_loader import ConfigLoader
```

**Step 2:** Replace hardcoded values

```python
# Before
imgsz = 416  # Hardcoded

# After
loader = ConfigLoader()
imgsz = loader.get('imgsz', default=416)
```

**Step 3:** Use priority chain for argparse

```python
# Before
parser.add_argument('--imgsz', type=int, default=640)
args = parser.parse_args()
imgsz = args.imgsz

# After
parser.add_argument('--imgsz', type=int)  # No default - let ConfigLoader handle it
args = parser.parse_args()
loader = ConfigLoader()
imgsz = loader.get('imgsz', cli_value=args.imgsz, default=416)
```

---

## Troubleshooting

### Problem: Configuration not applied

**Solution:** Check priority chain. Higher priority sources override lower ones.

```bash
# Debug mode shows which source was used
export RK_LOG_LEVEL=DEBUG
python scripts/run_rknn_sim.py --imgsz 640

# Output:
#   DEBUG: Config[imgsz] = 640 (source: CLI)
#   DEBUG: Config[conf_threshold] = 0.5 (source: YAML)
```

### Problem: Invalid configuration value

**Solution:** ConfigLoader validates types and ranges automatically.

```python
# Raises ValidationError with clear message
loader.get('imgsz', default=999, value_type=int, validate=validate_imgsz)
# ValidationError: Image size must be 416 or 640, got 999
```

### Problem: Configuration file not found

**Solution:** ConfigLoader gracefully falls back to defaults.

```python
loader = ConfigLoader(config_file='config/nonexistent.yaml')
# DEBUG: YAML config not found: config/nonexistent.yaml (using defaults)
```

---

## Best Practices

### 1. Use YAML for shared settings

```yaml
# config/app.yaml - Team shared configuration
imgsz: 416
conf_threshold: 0.5
```

### 2. Use ENV for deployment

```bash
# Docker, Kubernetes, systemd
Environment="RK_IMGSZ=640"
Environment="RK_LOG_LEVEL=INFO"
```

### 3. Use CLI for experiments

```bash
# Quick parameter sweep
for conf in 0.25 0.5 0.75; do
  python scripts/run_rknn_sim.py --conf $conf
done
```

### 4. Document priority chain

```python
# Good: Clear priority documentation
parser.add_argument('--imgsz', type=int, help='Image size (default: from config/app.yaml or 416)')

# Bad: Unclear default source
parser.add_argument('--imgsz', type=int, default=640)
```

### 5. Validate all inputs

```python
# Always use validation for critical parameters
loader.get('imgsz', value_type=int, validate=validate_imgsz)
```

---

## FAQ

**Q: Why can't I use argparse defaults anymore?**

A: Argparse defaults have the same priority as Python defaults, which breaks the priority chain. Use ConfigLoader defaults instead.

**Q: How do I set environment variables for all parameters?**

A: Use `RK_` prefix + uppercase key:
```bash
export RK_IMGSZ=640
export RK_CONF_THRESHOLD=0.5
export RK_TARGET_PLATFORM=rk3588
```

**Q: Can I use multiple YAML files?**

A: Yes, create different files for different environments:
```python
loader = ConfigLoader(config_file='config/production.yaml')
```

**Q: How do I know which source was used?**

A: Enable debug logging:
```bash
export RK_LOG_LEVEL=DEBUG
python scripts/run_rknn_sim.py
```

**Q: What happens if YAML has invalid syntax?**

A: ConfigLoader raises `ConfigurationError` with detailed message:
```
ConfigurationError: Invalid YAML in config/app.yaml: ...
```

---

## Summary

| Source | Priority | Use Case | Example |
|--------|----------|----------|---------|
| **CLI** | 1 (Highest) | One-off experiments | `--imgsz 640` |
| **ENV** | 2 | Docker/deployment | `RK_IMGSZ=640` |
| **YAML** | 3 | Shared team config | `imgsz: 640` |
| **Python** | 4 (Lowest) | Code defaults | `DEFAULT_SIZE = 416` |

**Golden Rule:** Always use ConfigLoader. Never hardcode configuration values.
