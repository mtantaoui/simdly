---
title: Performance Tips
description: Optimize your simdly code for maximum performance.
---

import { Card, CardGrid, Tabs, TabItem } from '@astrojs/starlight/components';

# Performance Optimization

<div class="performance-badge">
  Achieve up to 15x performance improvements
</div>

This guide covers essential techniques for maximizing performance with simdly.

## Performance Quick Wins

<CardGrid stagger>
  <Card title="🎯 Compiler Flags" icon="rocket">
    Enable AVX2 and link-time optimization for immediate 2-4x speedup
  </Card>
  <Card title="🧮 Memory Alignment" icon="setting">
    Use 32-byte aligned memory for optimal AVX2 vector performance
  </Card>
  <Card title="🔄 Loop Unrolling" icon="approve-check">
    Process multiple vectors per iteration to reduce loop overhead
  </Card>
  <Card title="📊 Data Layout" icon="puzzle">
    Structure of Arrays (SoA) layout optimizes SIMD operations
  </Card>
</CardGrid>

## Compilation Flags

### Essential Compiler Flags

Always enable AVX2 support for optimal performance:

<Tabs>
<TabItem label="Cargo.toml">

```toml
# Recommended approach - add to Cargo.toml
[build]
rustflags = ["-C", "target-feature=+avx2"]
```

</TabItem>
<TabItem label="Environment">

```bash
# Environment variable approach
export RUSTFLAGS="-C target-feature=+avx2"
cargo build --release
```

</TabItem>
<TabItem label="Command Line">

```bash
# Direct command line
cargo build --release -- -C target-feature=+avx2
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
# For modern Intel/AMD processors
RUSTFLAGS="-C target-cpu=native" cargo build --release

# For specific CPU features
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

## Memory Optimization

### Memory Alignment

Properly aligned memory provides significant performance benefits:

```rust
use std::alloc::{alloc, dealloc, Layout};
use simdly::simd::avx2::f32x8::F32x8;

// Allocate 32-byte aligned memory for optimal AVX2 performance
fn alloc_aligned_f32(count: usize) -> (*mut f32, Layout) {
    let layout = Layout::from_size_align(
        count * std::mem::size_of::<f32>(), 
        32  // AVX2 alignment requirement
    ).unwrap();
    
    let ptr = unsafe { alloc(layout) as *mut f32 };
    (ptr, layout)
}

// Example usage
let (aligned_ptr, layout) = alloc_aligned_f32(1024);

// Verify alignment
assert!(F32x8::is_aligned(aligned_ptr));

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
        for chunk in self.x.chunks_exact_mut(8) {
            let vec = F32x8::from_slice(chunk);
            // Process 8 x-coordinates simultaneously
            // ... SIMD operations ...
        }
    }
}
```

### Cache-Friendly Access Patterns

```rust
use simdly::simd::avx2::f32x8::F32x8;

// Cache-aware processing
fn cache_optimized_processing(data: &mut [f32]) {
    const CACHE_LINE_SIZE: usize = 64;  // bytes
    const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 4;  // 16 f32s
    
    // Process data in cache line chunks
    for cache_chunk in data.chunks_mut(FLOATS_PER_CACHE_LINE) {
        // Process each cache line with SIMD vectors
        for simd_chunk in cache_chunk.chunks_exact_mut(8) {
            let vec = F32x8::from_slice(simd_chunk);
            // ... SIMD operations ...
            unsafe {
                vec.store_unaligned_at(simd_chunk.as_mut_ptr());
            }
        }
        
        // Handle remainder in cache line
        let remainder = cache_chunk.len() % 8;
        if remainder > 0 {
            let start = cache_chunk.len() - remainder;
            let vec = F32x8::from_slice(&cache_chunk[start..]);
            // ... SIMD operations ...
        }
    }
}
```

## Algorithm Optimization

### Minimize Memory Operations

Prefer in-register operations over memory loads/stores:

```rust
// Good: Multiple operations on same vector
let vec = F32x8::from_slice(&data);
// Perform multiple operations without storing intermediate results
let result1 = vec; // operation 1
let result2 = result1; // operation 2  
let final_result = result2; // operation 3
// Store only final result

// Less efficient: Store/load between operations
let vec1 = F32x8::from_slice(&data);
// Store intermediate result
// Load again for next operation
// Store intermediate result again
// ...
```

### Loop Unrolling

Process multiple vectors per iteration:

```rust
use simdly::simd::avx2::f32x8::F32x8;

fn unrolled_processing(data: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    let mut i = 0;
    
    // Process 4 vectors (32 elements) per iteration
    while i + 32 <= data.len() {
        let vec1 = F32x8::from_slice(&data[i..i + 8]);
        let vec2 = F32x8::from_slice(&data[i + 8..i + 16]);
        let vec3 = F32x8::from_slice(&data[i + 16..i + 24]);
        let vec4 = F32x8::from_slice(&data[i + 24..i + 32]);
        
        // Process all 4 vectors
        // ... operations on vec1, vec2, vec3, vec4 ...
        
        // Store results
        unsafe {
            vec1.store_unaligned_at(result[i..].as_mut_ptr());
            vec2.store_unaligned_at(result[i + 8..].as_mut_ptr());
            vec3.store_unaligned_at(result[i + 16..].as_mut_ptr());
            vec4.store_unaligned_at(result[i + 24..].as_mut_ptr());
        }
        
        i += 32;
    }
    
    // Handle remaining elements
    while i + 8 <= data.len() {
        let vec = F32x8::from_slice(&data[i..i + 8]);
        // ... operations ...
        unsafe {
            vec.store_unaligned_at(result[i..].as_mut_ptr());
        }
        i += 8;
    }
    
    // Handle final partial vector
    if i < data.len() {
        let remaining = data.len() - i;
        let vec = unsafe { F32x8::load_partial(data[i..].as_ptr(), remaining) };
        // ... operations ...
        unsafe {
            vec.store_at_partial(result[i..].as_mut_ptr());
        }
    }
    
    result
}
```

### Avoid Branching in Hot Loops

Use conditional moves instead of branches:

```rust
// Less efficient: Branching in loop
for chunk in data.chunks_exact(8) {
    let vec = F32x8::from_slice(chunk);
    
    if some_condition {
        // Process one way
    } else {
        // Process another way
    }
}

// Better: Separate loops or use masking operations
if some_condition {
    for chunk in data.chunks_exact(8) {
        let vec = F32x8::from_slice(chunk);
        // Process one way
    }
} else {
    for chunk in data.chunks_exact(8) {
        let vec = F32x8::from_slice(chunk);
        // Process another way
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

### 1. Not Enabling AVX2
Always ensure AVX2 is enabled in your build configuration.

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
if F32x8::is_aligned(data.as_ptr()) {
    // Use aligned operations
} else {
    // Use unaligned operations or realign data
}
```

## Next Steps

- Review [SIMD Operations](/guides/simd-operations/) for implementation techniques
- Check the [API Reference](/reference/) for complete function documentation
- Profile your specific use cases to identify bottlenecks