# 📊 Simdly Benchmarking Guide

This guide explains how to run and interpret the comprehensive benchmarks for Simdly's addition operations across different algorithms and vector sizes.

## 🚀 Quick Start

```bash
# Run all addition benchmarks
cargo bench --bench addition
```

## 📈 Benchmark Suites

### 1. **addition.rs** - Comprehensive Addition Analysis

The main benchmark suite that provides complete performance analysis across:

- **Algorithm comparison**: Scalar vs SIMD vs Parallel implementations
- **Memory hierarchy**: L1 cache (4 KiB) → Main memory (128 MiB) 
- **Threshold analysis**: Fine-grained testing around performance transition points

```bash
# Run all addition benchmarks
cargo bench --bench addition

# Focus on specific size ranges
cargo bench --bench addition -- "Addition.*16_MiB"

# Test threshold analysis only
cargo bench --bench addition -- "Threshold_Analysis"

# Generate HTML report
cargo bench --bench addition -- --output-format html
```

**Key Metrics:**

- **Throughput (GiB/s)**: Memory bandwidth utilization
- **Elements/sec**: Pure computational throughput
- **Efficiency**: Actual vs theoretical speedup

### 2. **f32.rs** - Original Benchmark Suite

Existing benchmarks for mathematical operations including addition, cosine, absolute value, etc.

```bash
# Run original benchmarks
cargo bench --bench f32

# Focus on addition only
cargo bench --bench f32 -- "Addition"
```

## 📊 Understanding Results

### Performance Results

Latest benchmark results on test hardware (Linux x64, AVX2 enabled) with improved error handling:

| Vector Size | Scalar (GiB/s) | SIMD (GiB/s) | Parallel (GiB/s) | ndarray (GiB/s) | Best Algorithm |
|-------------|----------------|---------------|-------------------|------------------|----------------|
| 4 KiB       | 97.9          | 52.7          | N/A*             | 79.7            | Scalar         |
| 64 KiB      | 72.4          | 60.2          | 11.3             | 63.1            | Scalar         |
| 1 MiB       | 47.6          | 46.3          | 59.4             | 47.4            | Parallel       |
| 16 MiB      | 14.2          | 13.8          | 13.3             | 13.7            | Scalar         |
| 64 MiB      | 4.0           | 4.0           | 8.5              | 3.5             | Parallel       |

*Parallel not tested below 10,000 element threshold

### Performance Analysis

**Key Insights from Latest Benchmarks:**

- **Improved Error Handling Impact**: The new Result-based error handling shows minimal performance overhead
- **Scalar Performance**: Unexpectedly strong scalar performance, especially for smaller arrays
- **SIMD Effectiveness**: SIMD shows benefits primarily in the 1 MiB range
- **Parallel Scaling**: Parallel processing provides clear advantages for very large arrays (64+ MiB)
- **Memory Hierarchy**: Performance clearly degrades as we move from L1 cache to main memory

### Interpreting Results

**Key Observations:**

- **Small arrays (4-64 KiB)**: Scalar implementation performs exceptionally well, often outperforming SIMD due to lower overhead
- **Medium arrays (1 MiB)**: Parallel processing shows clear advantages with ~25% better throughput than scalar/SIMD
- **Large arrays (64+ MiB)**: Parallel implementation provides significant benefits with 2x throughput improvement

**Performance Insights:**

- **Error Handling Overhead**: Minimal performance impact from new Result-based error handling
- **Memory Hierarchy Effects**: Clear performance degradation as we move from cache to main memory
- **Algorithm Selection**: Different algorithms optimal at different scales:
  - **Scalar**: Best for small arrays due to minimal overhead
  - **SIMD**: Competitive but doesn't always beat scalar due to setup costs
  - **Parallel**: Clear winner for large arrays where thread overhead is justified
- **Compiler Optimizations**: Modern Rust compiler produces highly optimized scalar code that competes well with manual SIMD

**Recommendations:**

- Use scalar addition for arrays < 1 MiB unless specific SIMD benefits are measured
- Use parallel processing for arrays > 1 MiB to maximize throughput
- Always measure performance with your specific data patterns and hardware

## 🔧 Advanced Usage

### Custom Benchmark Configuration

```bash
# Longer measurement time for more accurate results
cargo bench --bench addition_rayon -- --measurement-time 60

# More samples for statistical significance
cargo bench --bench addition_rayon -- --sample-size 1000

# Specific size filtering
cargo bench --bench addition_rayon -- "Addition.*MiB"

# Output formats
cargo bench --bench addition_rayon -- --output-format csv
cargo bench --bench addition_rayon -- --output-format json
```

### Profiling Integration

```bash
# Profile with perf (Linux)
cargo bench --bench addition_rayon -- --profile-time=5

# Memory usage analysis
valgrind --tool=massif cargo bench --bench addition_rayon -- --quick

# CPU cache analysis  
perf stat -e cache-references,cache-misses,instructions,cycles \
  cargo bench --bench addition_rayon -- --quick
```

### Environment Variables

```bash
# Control Rayon thread count
RAYON_NUM_THREADS=8 cargo bench --bench rayon_scaling

# CPU affinity (Linux)
taskset -c 0-7 cargo bench --bench rayon_scaling

# Disable turbo boost for consistent results
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

## 📈 Performance Analysis Workflow

### 1. **Baseline Measurement**

```bash
# Establish baseline performance
cargo bench --bench addition -- "Addition.*4_KiB"
```

### 2. **Memory Hierarchy Analysis**

```bash
# Find cache transition points
cargo bench --bench addition -- "Addition"
```

### 3. **Algorithm Threshold Validation**

```bash
# Analyze performance transition points
cargo bench --bench addition -- "Threshold_Analysis"
```

## 🐛 Troubleshooting

### Common Issues

**Low Performance:**

- Check CPU frequency scaling: `cat /proc/cpuinfo | grep MHz`
- Verify AVX2 support: `lscpu | grep avx2`
- Ensure adequate memory: `free -h`

**Inconsistent Results:**

- Disable CPU frequency scaling: `sudo cpupower frequency-set --governor performance`
- Close unnecessary applications
- Use `--measurement-time` for longer averaging

**Benchmark Failures:**

- Insufficient memory for large vectors
- CPU feature not available (AVX2/NEON)
- Compiler optimization issues

### Performance Debugging

```bash
# Check CPU features
rustc --print cfg | grep target_feature

# Verify SIMD code generation
cargo rustc --release --bench addition -- --emit asm

# Algorithm threshold testing
cargo bench --bench addition -- "Threshold_Analysis"
```

## 📚 Further Reading

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [Rayon Documentation](https://docs.rs/rayon/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Programming](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)

## 🤝 Contributing

When adding new benchmarks:

1. **Follow naming conventions**: `benchmark_<category>_<specific_test>`
2. **Document thoroughly**: Add comprehensive function documentation
3. **Include multiple sizes**: Test across cache hierarchy
4. **Validate results**: Ensure correctness before performance
5. **Report metrics**: Use appropriate throughput measurements

Example benchmark structure:

```rust
/// Benchmarks [specific functionality].
///
/// Tests [what scenarios] across [what parameters].
/// Key metrics: [list main measurements].
fn benchmark_new_feature(c: &mut Criterion) {
    // Implementation
}
```
