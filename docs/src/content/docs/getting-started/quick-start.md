---
title: Quick Start Guide
description: Get up and running with Simdly cross-platform SIMD in minutes.
sidebar:
  order: 3
---

This guide will get you up and running with Simdly cross-platform SIMD in just a few minutes. Simdly automatically uses AVX2 on x86/x86_64 and NEON on ARM architectures.

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

### Platform-Specific SIMD Types

You can work directly with low-level SIMD vectors for more control:

#### x86_64 (AVX2) - F32x8 vectors

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdMath, SimdStore};

// Load data into SIMD vector
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = F32x8::from(data.as_slice());

// Basic math operations
let abs_result = vec.abs();           // Absolute value
let sqrt_result = vec.sqrt();         // Square root
let cos_result = vec.cos();          // Cosine

// Store results back to memory
let output = [0.0f32; 8];
cos_result.store_at(output.as_ptr());
```

#### aarch64 (NEON) - F32x4 vectors

```rust
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;
use simdly::simd::{SimdMath, SimdStore};

// Load data into SIMD vector
let data = [1.0, 2.0, 3.0, 4.0];
let vec = F32x4::from(data.as_slice());

// Basic math operations
let abs_result = vec.abs();           // Absolute value
let sqrt_result = vec.sqrt();         // Square root
let sin_result = vec.sin();          // Sine

// Store results back to memory
let output = [0.0f32; 4];
sin_result.store_at(output.as_ptr());
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
