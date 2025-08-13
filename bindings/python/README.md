# simdly

> ⚠️ **Development Status**: This project is currently under active development. Not yet published on PyPI.

High-performance SIMD mathematical operations for Python, powered by Rust.

## Overview

`simdly` provides Python bindings for the [simdly](https://github.com/mtantaoui/simdly) Rust library, offering:

- **SIMD-accelerated operations**: Leverages AVX2 (Intel) and NEON (ARM) instructions
- **Automatic optimization**: Intelligently selects the best algorithm based on data size
- **High performance**: Up to 13x speedup over scalar operations for mathematical functions
- **Memory efficient**: Optimized memory allocation and access patterns

## Installation

Build from source using maturin (not yet published on PyPI).

## Quick Start

The library provides element-wise operations with automatic SIMD optimization, including mathematical functions like cosine computation that leverage SIMD instructions for high performance on large arrays.

## Performance

The library automatically selects the optimal algorithm based on array size:

| Array Size | Strategy | Expected Speedup |
|------------|----------|------------------|
| < 256 elements | Scalar | 1x (minimal overhead) |
| 256 - 131K elements | SIMD | 4-8x |
| ≥ 131K elements | Parallel SIMD | 4-8x × CPU cores |

### Benchmark Results

Cosine computation performance on Intel AVX2:

| Array Size | Scalar (ms) | SIMD (ms) | **Speedup** |
|------------|-------------|-----------|-------------|
| 1K elements | 0.035 | 0.008 | **4.4x** |
| 64K elements | 1.4 | 0.12 | **11.7x** |
| 1M elements | 26 | 1.95 | **13.3x** |

## API Reference

### `add(a: List[float], b: List[float]) -> List[float]`

Element-wise addition of two float arrays.

**Parameters:**
- `a`: First input array
- `b`: Second input array (must be same length as `a`)

**Returns:**
- New array containing element-wise sum

**Raises:**
- `ValueError`: If arrays have different lengths or are empty

### `cos(a: List[float]) -> List[float]`

Element-wise cosine computation.

**Parameters:**
- `a`: Input array of angles in radians

**Returns:**
- New array containing cosine of each element

**Raises:**
- `ValueError`: If array is empty

## Architecture Support

- **Intel/AMD**: AVX2 (256-bit vectors) - Haswell (2013+) / Excavator (2015+)
- **ARM**: NEON (128-bit vectors) - ARMv8-A / Apple Silicon
- **Fallback**: Scalar operations on unsupported platforms

## Requirements

- Python 3.8+
- Supported CPU architecture (automatic fallback to scalar on unsupported platforms)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Contributing

Contributions welcome! See the main [simdly repository](https://github.com/mtantaoui/simdly) for development guidelines.