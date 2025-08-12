---
title: Quick Start
description: Get up and running with Simdly cross-platform SIMD in minutes.
sidebar:
  order: 3
---

# Quick Start Guide

This guide will get you up and running with Simdly cross-platform SIMD in just a few minutes. Simdly automatically uses AVX2 on x86/x86_64 and NEON on ARM architectures.

## Basic Usage

Here's a simple cross-platform example:

```rust
use simdly::simd::SimdMath;
use simdly::SimdAdd;

fn main() {
    // Create some data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    // Perform SIMD mathematical operations
    let doubled = data.iter().map(|x| x * 2.0).collect::<Vec<f32>>();
    let cosines = data.cos(); // SIMD accelerated cosine
    
    println!("Original: {:?}", data);
    println!("Doubled: {:?}", doubled);
    println!("Cosines: {:?}", cosines);
}
```

## Vector Operations

### High-Level SIMD Operations

```rust
use simdly::simd::SimdMath;

// Mathematical operations on vectors
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let results = data.cos(); // SIMD accelerated cosine
let sqrt_results = data.sqrt(); // SIMD accelerated square root
let exp_results = data.exp(); // SIMD accelerated exponential
```

### Vector Addition

```rust
use simdly::SimdAdd;

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];

// Choose algorithm based on data size:
let result = a.scalar_add(&b);    // For small arrays (< 128 elements)
let result = a.simd_add(&b);      // For medium arrays (128+ elements)
let result = a.par_simd_add(&b);  // For large arrays (262,144+ elements)
```

### Platform-Specific SIMD Types

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;

#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;

use simdly::simd::{SimdLoad, SimdStore};

#[cfg(target_arch = "x86_64")]
fn process_with_avx2() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = F32x8::from(&data[..]);
    
    let mut output = [0.0f32; 8];
    unsafe {
        vec.store_at(output.as_mut_ptr());
    }
}
```

## Mathematical Operations

### Basic Mathematical Functions

```rust
use simdly::simd::SimdMath;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Trigonometric functions
let cosines = data.cos();
let sines = data.sin();
let tangents = data.tan();

// Exponential and logarithmic
let exponentials = data.exp();
let logarithms = data.ln();
let square_roots = data.sqrt();

println!("Cosines: {:?}", cosines);
```

### Power and Distance Operations

```rust
use simdly::simd::SimdMath;

let base = vec![2.0, 3.0, 4.0, 5.0];
let exponent = vec![2.0, 2.0, 2.0, 2.0];
let powers = base.pow(exponent); // [4.0, 9.0, 16.0, 25.0]

let x = vec![3.0, 5.0, 8.0, 7.0];
let y = vec![4.0, 12.0, 15.0, 24.0];
let distances = x.hypot(y); // 2D Euclidean distance

println!("Powers: {:?}", powers);
println!("Distances: {:?}", distances);
```

## Algorithm Selection

Choose the right algorithm for your data size:

```rust
use simdly::SimdAdd;

fn choose_algorithm_example() {
    let small_data = vec![1.0; 100];  // < 128 elements
    let medium_data = vec![1.0; 5000]; // 128-262,143 elements
    let large_data = vec![1.0; 500_000]; // ≥ 262,144 elements
    
    let other = vec![2.0; 100];
    
    // For small arrays - scalar is fastest
    let result = small_data.scalar_add(&other);
    
    // For medium arrays - SIMD is optimal
    let medium_other = vec![2.0; 5000];
    let result = medium_data.simd_add(&medium_other);
    
    // For large arrays - parallel SIMD is best
    let large_other = vec![2.0; 500_000];
    let result = large_data.par_simd_add(&large_other);
}

## Performance Tips

1. **Choose the Right Algorithm**: Use `scalar_add()` for small arrays, `simd_add()` for medium arrays, and `par_simd_add()` for large arrays.

2. **Release Mode**: Always benchmark in release mode with optimizations enabled.

3. **Use SIMD Math**: Leverage `SimdMath` trait for mathematical operations on vectors.

4. **Parallel Processing**: Use parallel SIMD methods (`par_cos()`, `par_sin()`, etc.) for large datasets.

5. **Platform Detection**: The library automatically uses AVX2 on x86_64 and NEON on ARM.

6. **Compilation Flags**: For maximum performance, compile with: `RUSTFLAGS="-C target-feature=+avx2" cargo build --release`

## Next Steps

- Learn about [Cross-Platform SIMD Operations](/guides/simd-operations/) for more advanced usage
- Check out [Performance Tips](/guides/performance/) for optimization strategies across architectures
- Explore the [API Reference](/reference/) for detailed universal API documentation

## Common Patterns

### High-Level Vector Processing

```rust
use simdly::simd::SimdMath;

fn process_array(input: Vec<f32>) -> Vec<f32> {
    // Simple mathematical processing
    input.cos() // Automatically uses SIMD
}

fn complex_processing(input: Vec<f32>) -> Vec<f32> {
    let step1 = input.sin();
    let step2 = step1.exp();
    step2.sqrt()
}
```

### Algorithm Selection Pattern

```rust
use simdly::SimdAdd;

fn smart_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    match a.len() {
        0..128 => a.scalar_add(b),        // Small: use scalar
        128..262_144 => a.simd_add(b),    // Medium: use SIMD
        _ => a.par_simd_add(b),           // Large: use parallel SIMD
    }
}
```

### Platform-Specific Low-Level Operations

```rust
#[cfg(target_arch = "x86_64")]
fn low_level_processing() {
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdMath};
    
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = F32x8::from(&data[..]);
    let result = vec.sin(); // 8 parallel sine calculations
}
```