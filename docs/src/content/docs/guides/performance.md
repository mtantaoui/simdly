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

### Memory Alignment

Properly aligned memory provides significant performance benefits across platforms:

```rust
use std::alloc::{alloc, dealloc, Layout};
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{Alignment, SimdLoad, SimdStore};

// Platform-appropriate alignment
fn get_optimal_alignment() -> usize {
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        32  // AVX2 alignment (256-bit)
    } else if cfg!(target_arch = "aarch64") {
        16  // NEON alignment (128-bit) 
    } else {
        std::mem::align_of::<f32>()
    }
}

// Allocate optimally aligned memory
fn alloc_aligned_f32(count: usize) -> (*mut f32, Layout) {
    let alignment = get_optimal_alignment();
    let layout = Layout::from_size_align(
        count * std::mem::size_of::<f32>(),
        alignment
    ).unwrap();

    let ptr = unsafe { alloc(layout) as *mut f32 };
    (ptr, layout)
}

// Example usage (x86_64 with AVX2)
#[cfg(target_arch = "x86_64")]
{
    let (aligned_ptr, layout) = alloc_aligned_f32(1024);

    // Verify alignment
    assert!(F32x8::is_aligned(aligned_ptr));

    // Use aligned operations for best performance
    let vec = unsafe { F32x8::load_aligned(aligned_ptr) };
    // ... your SIMD operations ...
    unsafe { vec.store_aligned_at(aligned_ptr as *mut f32) };

    // Clean up
    unsafe { dealloc(aligned_ptr as *mut u8, layout) };
}
```

### Memory Layout Optimization

Organize data for sequential access patterns:

```rust
// Good: Array of Structures (AoS) for simple operations
struct Point3D {
    x: f32,
    y: f32,
    z: f32,
}

// Better: Structure of Arrays (SoA) for SIMD operations
struct Points3D {
    x: Vec<f32>,  // All x coordinates together
    y: Vec<f32>,  // All y coordinates together
    z: Vec<f32>,  // All z coordinates together
}

impl Points3D {
    #[cfg(target_arch = "x86_64")]
    fn process_x_coordinates(&mut self) {
        use simdly::simd::avx2::f32x8::F32x8;
        use simdly::simd::{SimdLoad, SimdStore};
        
        // AVX2: 8 f32 elements
        for chunk in self.x.chunks_exact_mut(8) {
            let vec = F32x8::from_slice(chunk);
            // Process coordinates using SIMD operations
            // ... SIMD operations ...
            unsafe { vec.store_unaligned_at(chunk.as_mut_ptr()) };
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn process_x_coordinates(&mut self) {
        // Use NEON implementation when available
        // Similar pattern with 4 f32 elements
        for chunk in self.x.chunks_exact_mut(4) {
            // ... NEON SIMD operations ...
        }
    }
}
```

### Cache-Friendly Access Patterns

```rust
#[cfg(target_arch = "x86_64")]
fn cache_optimized_processing(data: &mut [f32]) {
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore};
    
    const CACHE_LINE_SIZE: usize = 64;  // bytes
    const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 4;  // 16 f32s
    const VECTOR_SIZE: usize = 8;  // AVX2 processes 8 f32s

    // Process data in cache line chunks
    for cache_chunk in data.chunks_mut(FLOATS_PER_CACHE_LINE) {
        // Process each cache line with SIMD vectors
        for simd_chunk in cache_chunk.chunks_exact_mut(VECTOR_SIZE) {
            let vec = F32x8::from_slice(simd_chunk);
            // ... SIMD operations ...
            unsafe { vec.store_unaligned_at(simd_chunk.as_mut_ptr()) };
        }

        // Handle remainder in cache line
        let remainder = cache_chunk.len() % VECTOR_SIZE;
        if remainder > 0 {
            let start = cache_chunk.len() - remainder;
            let vec = unsafe { F32x8::load_partial(cache_chunk[start..].as_ptr(), remainder) };
            // ... SIMD operations ...
            unsafe { vec.store_at_partial(cache_chunk[start..].as_mut_ptr()) };
        }
    }
}
```

## Algorithm Optimization

### Minimize Memory Operations

Prefer in-register operations over memory loads/stores:

```rust
// Good: Multiple operations on same vector
#[cfg(target_arch = "x86_64")]
{
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore, SimdMath};
    use std::ops::Add;
    
    let vec = unsafe { F32x8::load_unaligned(data.as_ptr()) };
    let other_vec = unsafe { F32x8::load_unaligned(other_data.as_ptr()) };
    
    // Perform multiple operations without storing intermediate results
    let result1 = vec.add(other_vec);    // operation 1
    let result2 = result1.abs();         // operation 2 (using SimdMath trait)
    // Store only final result
    unsafe { result2.store_unaligned_at(output.as_mut_ptr()) };

    // Less efficient: Store/load between operations
    let vec1 = unsafe { F32x8::load_unaligned(data.as_ptr()) };
    unsafe { vec1.store_unaligned_at(temp.as_mut_ptr()) };     // Store intermediate result
    let vec2 = unsafe { F32x8::load_unaligned(temp.as_ptr()) };  // Load again for next operation
    // ... inefficient memory operations
}
```

### Loop Unrolling

Process multiple vectors per iteration:

```rust
#[cfg(target_arch = "x86_64")]
fn unrolled_processing(data: &[f32], scale: f32) -> Vec<f32> {
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore};
    
    let mut result = vec![0.0; data.len()];
    let mut i = 0;
    
    const LANE_COUNT: usize = 8;  // AVX2 processes 8 f32s
    const UNROLL_FACTOR: usize = 4;
    const UNROLL_SIZE: usize = LANE_COUNT * UNROLL_FACTOR;

    // Process 4 vectors per iteration
    while i + UNROLL_SIZE <= data.len() {
        let vec1 = unsafe { F32x8::load_unaligned(data[i..].as_ptr()) };
        let vec2 = unsafe { F32x8::load_unaligned(data[i + LANE_COUNT..].as_ptr()) };
        let vec3 = unsafe { F32x8::load_unaligned(data[i + LANE_COUNT * 2..].as_ptr()) };
        let vec4 = unsafe { F32x8::load_unaligned(data[i + LANE_COUNT * 3..].as_ptr()) };

        // Process all 4 vectors - multiply by scale factor
        // Note: Would need to implement scalar multiplication in your crate
        // For now showing the pattern with existing operations
        
        // Store results
        unsafe {
            vec1.store_unaligned_at(result[i..].as_mut_ptr());
            vec2.store_unaligned_at(result[i + LANE_COUNT..].as_mut_ptr());
            vec3.store_unaligned_at(result[i + LANE_COUNT * 2..].as_mut_ptr());
            vec4.store_unaligned_at(result[i + LANE_COUNT * 3..].as_mut_ptr());
        }

        i += UNROLL_SIZE;
    }

    // Handle remaining full vectors
    while i + LANE_COUNT <= data.len() {
        let vec = unsafe { F32x8::load_unaligned(data[i..].as_ptr()) };
        unsafe { vec.store_unaligned_at(result[i..].as_mut_ptr()) };
        i += LANE_COUNT;
    }

    // Handle final partial vector
    if i < data.len() {
        let remaining = data.len() - i;
        let vec = unsafe { F32x8::load_partial(data[i..].as_ptr(), remaining) };
        unsafe { vec.store_at_partial(result[i..].as_mut_ptr()) };
    }

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
    
    // Less efficient: Branching in loop
    for chunk in data.chunks_exact(LANE_COUNT) {
        let vec = F32x8::from_slice(chunk);

        if some_condition {
            // Process one way
        } else {
            // Process another way
        }
    }

    // Better: Separate loops or use masking operations
    if some_condition {
        for chunk in data.chunks_exact_mut(LANE_COUNT) {
            let vec = F32x8::from_slice(chunk);
            let offset_vec = F32x8::from_slice(&offset_data);
            
            // Process one way - add operation
            let result = vec.add(offset_vec);
            unsafe { result.store_unaligned_at(chunk.as_mut_ptr()) };
        }
    } else {
        for chunk in data.chunks_exact_mut(LANE_COUNT) {
            let vec = F32x8::from_slice(chunk);
            // Process another way - would need scalar multiplication implemented
            unsafe { vec.store_unaligned_at(chunk.as_mut_ptr()) };
        }
    }
}
```

## Large Data Processing

### Efficient Batch Processing

For large datasets, focus on efficient chunking and memory access patterns:

```rust
#[cfg(target_arch = "x86_64")]
fn process_large_dataset(input: &[f32], output: &mut [f32]) {
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore, Alignment};
    
    assert_eq!(input.len(), output.len());

    // Process in chunks of 8 elements (AVX2 vector size)
    for (input_chunk, output_chunk) in input.chunks_exact(8).zip(output.chunks_exact_mut(8)) {
        let vec = F32x8::from_slice(input_chunk);

        // ... your SIMD operations ...

        // Store results
        unsafe {
            vec.store_unaligned_at(output_chunk.as_mut_ptr());
        }
    }

    // Handle remainder with scalar operations
    let remainder_start = (input.len() / 8) * 8;
    if remainder_start < input.len() {
        for i in remainder_start..input.len() {
            output[i] = input[i]; // Your scalar operation here
        }
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
// benches/simd_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdLoad;

fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();

        group.bench_with_input(
            BenchmarkId::new("load_store", size),
            &data,
            |b, data| {
                b.iter(|| {
                    for chunk in data.chunks_exact(8) {
                        let vec = F32x8::from_slice(chunk);
                        // Simulate some operation
                        std::hint::black_box(vec);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_simd_operations);
criterion_main!(benches);
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
// Check alignment when performance is critical
#[cfg(target_arch = "x86_64")]
{
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{Alignment, SimdLoad};
    
    if F32x8::is_aligned(data.as_ptr()) {
        // Use aligned operations
        let vec = unsafe { F32x8::load_aligned(data.as_ptr()) };
    } else {
        // Use unaligned operations or realign data
        let vec = unsafe { F32x8::load_unaligned(data.as_ptr()) };
    }
}
```

## Next Steps

- Review [SIMD Operations](/guides/simd-operations/) for implementation techniques
- Check the [API Reference](/reference/) for complete function documentation
- Profile your specific use cases to identify bottlenecks
