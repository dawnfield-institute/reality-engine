# Scripts

Utility scripts for testing and benchmarking Reality Engine.

## Available Scripts

### `benchmark_gpu.py`
Performance testing for GPU operations.

```bash
python scripts/benchmark_gpu.py
```

Tests:
- Field initialization speed
- Laplacian computation
- Herniation detection
- Full evolution step timing

### `test_gpu.py`
Verify CUDA availability and basic operations.

```bash
python scripts/test_gpu.py
```

Checks:
- PyTorch installation
- CUDA availability
- GPU device info
- Basic tensor operations

### `debug_herniations.py`
Debug herniation cascade dynamics.

```bash
python scripts/debug_herniations.py
```

Analyzes:
- Herniation site selection
- Cascade propagation
- Field pressure buildup
- I→M transfer mechanics

## Usage

Run scripts directly from project root:

```bash
# Benchmark
python scripts/benchmark_gpu.py

# Test GPU
python scripts/test_gpu.py

# Debug
python scripts/debug_herniations.py
```

## Requirements

All scripts use the main `requirements.txt` dependencies.
