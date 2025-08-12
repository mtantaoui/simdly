---
title: Performance Tips
description: Optimize your Simdly code for maximum performance across x86 and ARM architectures.
---

# Cross-Platform Performance Optimization

<div class="performance-badge">
  Optimize your SIMD code with robust error handling
</div>

This guide covers essential techniques for maximizing performance with Simdly across both x86 (AVX2) and ARM (NEON) architectures, including real benchmark results and error handling capabilities.

## Latest Benchmark Results

Performance measurements on Linux x64 with AVX2 and improved error handling:

| Vector Size | Scalar (GiB/s) | SIMD (GiB/s) | ndarray (GiB/s) | Parallel (GiB/s) | Best Algorithm |
|-------------|----------------|---------------|------------------|-------------------|----------------|
| 4 KiB       | 67.0           | 40.4          | 54.2             | N/A*             | Scalar         |
| 16 KiB      | 54.3           | 36.5          | 44.2             | N/A*             | Scalar         |
| 64 KiB      | 47.6           | 48.1          | 23.3             | 2.0              | SIMD           |
| 256 KiB     | 34.4           | 30.4          | 34.0             | 6.0              | Scalar         |
| 1 MiB       | 34.2           | 32.8          | 34.1             | 11.9             | Scalar         |
| 4 MiB       | 18.4           | 17.9          | 18.3             | 18.2             | Scalar         |
| 16 MiB      | 12.9           | 12.7          | 12.8             | 12.6             | Scalar         |
| 64 MiB      | 3.6            | 3.6           | 3.2              | 8.4              | Parallel       |
| 128 MiB     | 3.2            | 3.0           | 3.3              | 8.1              | Parallel       |

*Parallel processing shows overhead below 64 KiB threshold

### Key Performance Insights

- **Scalar Dominance**: Scalar code outperforms SIMD for small vectors (≤16 KiB) due to minimal overhead
- **SIMD Sweet Spot**: Best performance at 64 KiB where SIMD marginally wins (48.1 vs 47.6 GiB/s)
- **Memory Hierarchy Effects**: Clear degradation as data exceeds cache levels (67.0 → 3.2 GiB/s)
- **Algorithm Selection**: Each approach has optimal ranges - scalar for small/medium, parallel for very large
- **Parallel Benefits**: Only worthwhile for very large datasets (≥64 MiB) where parallel achieves 8+ GiB/s
- **ndarray Performance**: Competitive with scalar for most sizes, showing good optimization

## Performance Quick Wins

<CardGrid stagger>
  <Card title="🎯 Compiler Flags" icon="rocket">
    Enable target-specific optimizations: AVX2 on x86, NEON on ARM for immediate 2-4x speedup
  </Card>
  <Card title="🧮 Memory Alignment" icon="setting">
    Use 32-byte alignment (x86) or 16-byte alignment (ARM) for optimal vector performance
  </Card>
  <Card title="🔄 Loop Unrolling" icon="approve-check">
    Process multiple vectors per iteration to reduce loop overhead across platforms
  </Card>
  <Card title="📊 Data Layout" icon="puzzle">
    Structure of Arrays (SoA) layout optimizes SIMD operations on both architectures
  </Card>
</CardGrid>

## Compilation Flags

### Essential Compiler Flags

Simdly comes pre-configured with optimal SIMD settings via `.cargo/config.toml` with `target-cpu=native`, but you can customize for specific scenarios:

<Tabs>
<TabItem label="Default (Recommended)">

```toml
# Simdly's built-in .cargo/config.toml (no action needed!)
[build]
rustflags = ["-C", "target-cpu=native"]
```

```bash
# Just build - automatic optimization included!
cargo build --release
```

</TabItem>
<TabItem label="Custom x86/x86_64">

```bash
# Force specific AVX2 features
export RUSTFLAGS="-C target-feature=+avx2,+fma"
cargo build --release
```

</TabItem>
<TabItem label="Custom ARM">

```bash
# Force specific NEON features
export RUSTFLAGS="-C target-feature=+neon,+fp-armv8"
cargo build --release
```

</TabItem>
</Tabs>

### Advanced Optimization Flags

For maximum performance in release builds:

```toml
# Add to Cargo.toml
[profile.release]
lto = "fat"              # Link-time optimization
codegen-units = 1        # Better optimization  
panic = "abort"          # Smaller binaries, no unwinding
opt-level = 3            # Maximum optimization
```

### Target-Specific Compilation

Compile for your specific CPU architecture:

```bash
# Default: Simdly automatically uses target-cpu=native
cargo build --release

# Override for specific features (advanced)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release      # x86
RUSTFLAGS="-C target-feature=+neon,+fp-armv8" cargo build --release # ARM

# Cross-compilation examples (automatic feature detection)
cargo build --release --target aarch64-unknown-linux-gnu  # ARM64 Linux
cargo build --release --target x86_64-pc-windows-msvc     # x86_64 Windows
```

## Memory Optimization

### Algorithm Selection for Performance

Choose the right algorithm based on your data size for optimal performance:

```rust
use simdly::SimdAdd;
use simdly::simd::SimdMath;

// Smart algorithm selection based on data size
fn optimized_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    match a.len() {
        // Small arrays: scalar is fastest (avoids SIMD overhead)
        0..128 => a.scalar_add(b),
        
        // Medium arrays: SIMD is optimal
        128..262_144 => a.simd_add(b),
        
        // Large arrays: parallel SIMD is best
        _ => a.par_simd_add(b),
    }
}

// Mathematical operations automatically use SIMD
fn math_operations_example(data: Vec<f32>) -> Vec<f32> {
    // High-level SIMD operations
    let step1 = data.cos();      // SIMD accelerated cosine
    let step2 = step1.exp();     // SIMD accelerated exponential
    step2.sqrt()                 // SIMD accelerated square root
}
```

### High-Level SIMD Usage

Leverage the high-level SIMD traits for optimal performance:

```rust
use simdly::simd::SimdMath;
use simdly::SimdAdd;

// Structure of Arrays (SoA) for SIMD operations
struct Points3D {
    x: Vec<f32>,  // All x coordinates together
    y: Vec<f32>,  // All y coordinates together
    z: Vec<f32>,  // All z coordinates together
}

impl Points3D {
    fn process_coordinates(&mut self) {
        // Use high-level SIMD operations automatically
        self.x = self.x.cos();        // SIMD accelerated cosine
        self.y = self.y.sin();        // SIMD accelerated sine
        self.z = self.z.sqrt();       // SIMD accelerated square root
    }
    
    fn add_offset(&mut self, offset: f32) {
        let offset_vec = vec![offset; self.x.len()];
        
        // Choose algorithm based on data size
        self.x = match self.x.len() {
            0..128 => self.x.scalar_add(&offset_vec),
            128..262_144 => self.x.simd_add(&offset_vec),
            _ => self.x.par_simd_add(&offset_vec),
        };
    }
    
    fn compute_distances(&self) -> Vec<f32> {
        // 2D distance calculation using SIMD
        self.x.hypot(&self.y)
    }
}
```

### Parallel Processing for Large Datasets

```rust
use simdly::simd::SimdMath;

fn process_large_dataset(data: Vec<f32>) -> Vec<f32> {
    // Parallel methods automatically handle PARALLEL_SIMD_THRESHOLD!
    // No manual threshold checking needed
    data.par_cos()  // Automatically uses parallel for arrays ≥ 262,144 elements
                    // Automatically uses single-threaded SIMD for smaller arrays
}

// Chain multiple operations efficiently with automatic thresholds
fn complex_mathematical_pipeline(data: Vec<f32>) -> Vec<f32> {
    // All par_ methods automatically handle thresholds - much simpler!
    data.par_sin()       // Automatically chooses parallel vs single-threaded
        .par_exp()       // Each method handles its own threshold
        .par_sqrt()      // Clean and efficient
}
```

## Algorithm Optimization

### Minimize Memory Operations

Prefer in-register operations over memory loads/stores:

```rust
// Good: Chain multiple high-level operations
use simdly::simd::SimdMath;
use simdly::SimdAdd;

// Efficient operation chaining
fn efficient_operations(data: Vec<f32>, other: Vec<f32>) -> Vec<f32> {
    // Chain operations efficiently
    let result1 = data.simd_add(&other);  // SIMD addition
    let result2 = result1.abs();          // SIMD absolute value
    result2.sqrt()                        // SIMD square root
}

// Less efficient: Separate operations
fn less_efficient_operations(data: Vec<f32>, other: Vec<f32>) -> Vec<f32> {
    let temp1 = data.simd_add(&other);    // Store intermediate result
    let temp2 = temp1.abs();              // Another intermediate result
    temp2.sqrt()                          // Final result
}
```

### Loop Unrolling

Process multiple vectors per iteration:

```rust
fn batch_processing_example(data: Vec<f32>) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    // Use appropriate algorithm based on data size
    if data.len() >= 262_144 {
        // Large data: use parallel SIMD for maximum throughput
        data.par_cos()
    } else {
        // Smaller data: use regular SIMD
        data.cos()
    }
}

// Complex mathematical processing
fn complex_batch_processing(data: Vec<f32>) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    // Chain multiple operations efficiently
    let result = data
        .sin()      // Step 1: SIMD sine
        .abs()      // Step 2: SIMD absolute value
        .sqrt();    // Step 3: SIMD square root
    
    result
}
```

### Avoid Branching in Hot Loops

Use conditional moves instead of branches:

```rust
#[cfg(target_arch = "x86_64")]
{
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore};
    use std::ops::Add;
    
    const LANE_COUNT: usize = 8;
    
    // Avoid branching by using conditional processing
fn conditional_processing(data: Vec<f32>, use_cosine: bool) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    // Better: Process the entire array with one operation
    if use_cosine {
        data.cos()  // Process entire array with cosine
    } else {
        data.sin()  // Process entire array with sine
    }
}

// Example with multiple operations
fn complex_conditional_processing(data: Vec<f32>, operation_type: u32) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    match operation_type {
        0 => data.cos().abs(),           // Cosine then absolute value
        1 => data.sin().sqrt(),          // Sine then square root
        2 => data.exp().ln(),            // Exponential then natural log
        _ => data,                       // No operation
    }
}
}
```

## Large Data Processing

### Efficient Batch Processing

For large datasets, focus on efficient chunking and memory access patterns:

```rust
fn process_large_dataset_auto(input: Vec<f32>) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    // Simplest approach: let par_cos handle everything automatically
    input.par_cos()  // Automatically chooses optimal strategy
}

// If you want manual control over algorithm selection:
fn process_large_dataset_manual(input: Vec<f32>) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    match input.len() {
        // Use parallel SIMD for large datasets
        262_144.. => input.par_cos(),
        
        // Use regular SIMD for medium datasets
        128.. => input.cos(),
        
        // Use scalar for small datasets
        _ => input.into_iter().map(|x| x.cos()).collect(),
    }
}

// Multi-step processing pipeline
fn processing_pipeline(input: Vec<f32>) -> Vec<f32> {
    use simdly::simd::SimdMath;
    
    let step1 = if input.len() >= 262_144 {
        input.par_sin()     // Parallel SIMD sine
    } else {
        input.sin()         // Regular SIMD sine
    };
    
    let step2 = if step1.len() >= 262_144 {
        step1.par_abs()     // Parallel SIMD absolute value
    } else {
        step1.abs()         // Regular SIMD absolute value
    };
    
    if step2.len() >= 262_144 {
        step2.par_sqrt()    // Parallel SIMD square root
    } else {
        step2.sqrt()        // Regular SIMD square root
    }
}
```

## Profiling and Benchmarking

### Benchmarking Setup

Use `criterion` for accurate benchmarking:

```toml
# Add to Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "simd_benchmarks"
harness = false
```

```rust
The simdly crate includes comprehensive benchmarks that demonstrate real performance improvements.

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# View detailed results in your browser
open target/criterion/report/index.html
```

### Available Benchmarks

The crate includes benchmarks for:
- Mathematical operations (`cos`, `sin`, `exp`, `sqrt`, `ln`, etc.)
- Vector addition with different algorithms (`scalar_add`, `simd_add`, `par_simd_add`)
- Comparison operations
- Various array sizes to demonstrate threshold effects

### Understanding Results

Benchmark results show:
- **Actual speedup** on your specific hardware
- **Threshold effects** where parallel methods switch strategies
- **Memory bandwidth** limitations for very large arrays
- **Platform differences** between AVX2 and NEON architectures

*Note: Performance varies significantly by CPU architecture, so always benchmark on your target hardware.*
```

### Performance Monitoring

Use CPU performance counters to understand bottlenecks:

```bash
# Profile cache misses
perf stat -e cache-misses,cache-references cargo bench

# Profile instruction metrics
perf stat -e instructions,cycles cargo bench

# Generate detailed reports
perf record cargo bench
perf report
```

## Common Performance Pitfalls

### 1. Not Enabling Target-Specific SIMD

Always ensure proper SIMD features are enabled for your target platform:

- AVX2 for x86/x86_64
- NEON for ARM/AArch64

### 2. Mixed Data Types

Keep data types consistent to avoid expensive conversions:

```rust
// Good: All f32
let data: Vec<f32> = vec![1.0, 2.0, 3.0];

// Less efficient: Mixed types requiring conversion
let mixed: Vec<f64> = vec![1.0, 2.0, 3.0];  // f64 -> f32 conversion needed
```

### 3. Frequent Small Allocations

Pre-allocate buffers when possible:

```rust
// Good: Pre-allocated buffer
let mut buffer = vec![0.0f32; 1024];
for chunk in input.chunks(8) {
    // Reuse buffer
}

// Bad: Frequent allocations
for chunk in input.chunks(8) {
    let temp = vec![0.0f32; 8];  // Allocation per iteration
}
```

### 4. Ignoring Memory Alignment

Always consider alignment for optimal performance:

```rust
// Use high-level APIs that handle performance automatically
use simdly::simd::SimdMath;
use simdly::SimdAdd;

// Good: High-level API handles optimization automatically
let result = data.cos();           // Automatically optimized
let sum = a.simd_add(&b);          // Automatically optimized

// Algorithm selection based on data size
let smart_result = match data.len() {
    0..128 => data.iter().map(|x| x.cos()).collect::<Vec<f32>>(),  // Scalar
    128..262_144 => data.cos(),                                     // SIMD
    _ => data.par_cos(),                                            // Parallel SIMD
};
```

## Next Steps

- Review [SIMD Operations](/guides/simd-operations/) for implementation techniques
- Check the [API Reference](/reference/) for complete function documentation
- Profile your specific use cases to identify bottlenecks
