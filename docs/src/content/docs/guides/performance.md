---
title: Performance Tips
description: Optimize your Simdly code for maximum performance across x86 and ARM architectures.
---

# Cross-Platform Performance Optimization

<div class="performance-badge">
  Achieve up to 15x performance improvements
</div>

This guide covers essential techniques for maximizing performance with Simdly across both x86 (AVX2) and ARM (NEON) architectures.

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

For maximum performance in production:

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
use simdly::f32x8;  // Works on both AVX2 and NEON

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

// Example usage
let (aligned_ptr, layout) = alloc_aligned_f32(1024);

// Verify alignment
assert!(f32x8::is_aligned(aligned_ptr));

// Use aligned operations for best performance
// ... your SIMD operations ...

// Clean up
unsafe { dealloc(aligned_ptr as *mut u8, layout) };
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
    fn process_x_coordinates(&mut self) {
        // Universal chunk size that works on both platforms
        let chunk_size = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
            8  // AVX2: 8 f32 elements
        } else {
            4  // NEON: 4 f32 elements
        };
        
        for chunk in self.x.chunks_exact_mut(chunk_size) {
            let vec = f32x8::load_unaligned(chunk);
            // Process coordinates using universal API
            // ... SIMD operations ...
            vec.store_unaligned(chunk);
        }
    }
}
```

### Cache-Friendly Access Patterns

```rust
use simdly::f32x8;

// Cache-aware processing with cross-platform optimization
fn cache_optimized_processing(data: &mut [f32]) {
    const CACHE_LINE_SIZE: usize = 64;  // bytes (typical for both x86 and ARM)
    const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 4;  // 16 f32s
    
    // Platform-specific vector size
    let vector_size = f32x8::lanes();

    // Process data in cache line chunks
    for cache_chunk in data.chunks_mut(FLOATS_PER_CACHE_LINE) {
        // Process each cache line with SIMD vectors
        for simd_chunk in cache_chunk.chunks_exact_mut(vector_size) {
            let vec = f32x8::load_unaligned(simd_chunk);
            // ... SIMD operations ...
            vec.store_unaligned(simd_chunk);
        }

        // Handle remainder in cache line
        let remainder = cache_chunk.len() % vector_size;
        if remainder > 0 {
            let start = cache_chunk.len() - remainder;
            let vec = f32x8::load_partial(&cache_chunk[start..], remainder);
            // ... SIMD operations ...
            vec.store_partial(&mut cache_chunk[start..], remainder);
        }
    }
}
```

## Algorithm Optimization

### Minimize Memory Operations

Prefer in-register operations over memory loads/stores:

```rust
// Good: Multiple operations on same vector (works on both platforms)
let vec = f32x8::load_unaligned(&data);
// Perform multiple operations without storing intermediate results
let result1 = vec.add(other_vec);    // operation 1
let result2 = result1.mul(scale);    // operation 2
let final_result = result2.abs();    // operation 3
// Store only final result
final_result.store_unaligned(&mut output);

// Less efficient: Store/load between operations
let vec1 = f32x8::load_unaligned(&data);
vec1.store_unaligned(&mut temp);     // Store intermediate result
let vec2 = f32x8::load_unaligned(&temp);  // Load again for next operation
// ... inefficient memory operations
```

### Loop Unrolling

Process multiple vectors per iteration:

```rust
use simdly::f32x8;

fn unrolled_processing(data: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    let mut i = 0;
    
    // Platform-specific vector size
    let lane_count = f32x8::lanes();
    let unroll_factor = 4;
    let unroll_size = lane_count * unroll_factor;

    // Process 4 vectors per iteration (adaptive to platform)
    while i + unroll_size <= data.len() {
        let vec1 = f32x8::load_unaligned(&data[i..]);
        let vec2 = f32x8::load_unaligned(&data[i + lane_count..]);
        let vec3 = f32x8::load_unaligned(&data[i + lane_count * 2..]);
        let vec4 = f32x8::load_unaligned(&data[i + lane_count * 3..]);

        // Process all 4 vectors (operations work on both AVX2 and NEON)
        let result1 = vec1.mul(scale);  // example operation
        let result2 = vec2.mul(scale);
        let result3 = vec3.mul(scale);
        let result4 = vec4.mul(scale);

        // Store results
        result1.store_unaligned(&mut result[i..]);
        result2.store_unaligned(&mut result[i + lane_count..]);
        result3.store_unaligned(&mut result[i + lane_count * 2..]);
        result4.store_unaligned(&mut result[i + lane_count * 3..]);

        i += unroll_size;
    }

    // Handle remaining full vectors
    while i + lane_count <= data.len() {
        let vec = f32x8::load_unaligned(&data[i..]);
        let processed = vec.mul(scale);  // example operation
        processed.store_unaligned(&mut result[i..]);
        i += lane_count;
    }

    // Handle final partial vector
    if i < data.len() {
        let remaining = data.len() - i;
        let vec = f32x8::load_partial(&data[i..], remaining);
        let processed = vec.mul(scale);  // example operation
        processed.store_partial(&mut result[i..], remaining);
    }

    result
}
```

### Avoid Branching in Hot Loops

Use conditional moves instead of branches:

```rust
// Less efficient: Branching in loop (same issue on both platforms)
let lane_count = f32x8::lanes();
for chunk in data.chunks_exact(lane_count) {
    let vec = f32x8::load_unaligned(chunk);

    if some_condition {
        // Process one way
    } else {
        // Process another way
    }
}

// Better: Separate loops or use masking operations
if some_condition {
    for chunk in data.chunks_exact(lane_count) {
        let vec = f32x8::load_unaligned(chunk);
        // Process one way using universal API
        let result = vec.add(offset);
        result.store_unaligned(chunk);
    }
} else {
    for chunk in data.chunks_exact(lane_count) {
        let vec = f32x8::load_unaligned(chunk);
        // Process another way using universal API
        let result = vec.mul(factor);
        result.store_unaligned(chunk);
    }
}
```

## Streaming and Large Data

### Non-Temporal Stores

For large datasets that won't be reused:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdStore;

fn process_large_dataset(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    // Ensure output is properly aligned for streaming
    let output_ptr = output.as_mut_ptr();
    assert!(F32x8::is_aligned(output_ptr));

    for (i, chunk) in input.chunks_exact(8).enumerate() {
        let vec = F32x8::from_slice(chunk);

        // ... your operations ...

        // Use streaming store for large data (bypasses cache)
        unsafe {
            vec.stream_at(output_ptr.add(i * 8));
        }
    }

    // Handle remainder with regular stores
    let remainder_start = (input.len() / 8) * 8;
    if remainder_start < input.len() {
        let remainder_size = input.len() - remainder_start;
        let vec = unsafe {
            F32x8::load_partial(input[remainder_start..].as_ptr(), remainder_size)
        };

        // ... your operations ...

        unsafe {
            vec.store_at_partial(output[remainder_start..].as_mut_ptr());
        }
    }
}
```

### Prefetching

For predictable access patterns, consider prefetching:

```rust
use std::arch::x86_64::_mm_prefetch;

fn prefetch_example(data: &[f32]) {
    const PREFETCH_DISTANCE: usize = 64; // Elements ahead to prefetch

    for (i, chunk) in data.chunks_exact(8).enumerate() {
        // Prefetch data that will be needed soon
        if i * 8 + PREFETCH_DISTANCE < data.len() {
            unsafe {
                _mm_prefetch(
                    data[i * 8 + PREFETCH_DISTANCE..].as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0  // Prefetch to L1 cache
                );
            }
        }

        let vec = F32x8::from_slice(chunk);
        // ... process vector ...
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
// Check alignment when performance is critical (universal API)
if f32x8::is_aligned(data.as_ptr()) {
    // Use aligned operations
    let vec = f32x8::load_aligned(data);
} else {
    // Use unaligned operations or realign data
    let vec = f32x8::load_unaligned(data);
}
```

## Next Steps

- Review [SIMD Operations](/guides/simd-operations/) for implementation techniques
- Check the [API Reference](/reference/) for complete function documentation
- Profile your specific use cases to identify bottlenecks
