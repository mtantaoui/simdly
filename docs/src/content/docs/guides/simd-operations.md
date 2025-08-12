---
title: SIMD Operations
description: Learn about SIMD operations and optimization techniques with simdly.
---

# SIMD Operations

This guide covers SIMD operations available in simdly and how to use them effectively.

## Understanding SIMD

SIMD (Single Instruction, Multiple Data) allows you to perform the same operation on multiple data elements simultaneously. Simdly automatically uses AVX2 on x86_64 (8 f32 values) and NEON on ARM (4 f32 values) for optimal performance.

## Core Operations

### High-Level Mathematical Operations

simdly provides high-level mathematical operations that automatically use SIMD:

#### Basic Mathematical Functions

```rust
use simdly::simd::SimdMath;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Trigonometric functions (automatically SIMD accelerated)
let cosines = data.cos();
let sines = data.sin();
let tangents = data.tan();

// Exponential and logarithmic functions
let exponentials = data.exp();
let logarithms = data.ln();
let square_roots = data.sqrt();

println!("Cosines: {:?}", cosines);
```

#### Vector Addition Operations

Choose the right algorithm based on your data size:

```rust
use simdly::SimdAdd;

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];

// Choose algorithm based on data size:
let result = a.scalar_add(&b);      // For small arrays (< 128 elements)
let result = a.simd_add(&b);        // For medium arrays (128+ elements)
let result = a.par_simd_add(&b);    // For large arrays (262,144+ elements)
```

### Power and Distance Operations

Advanced mathematical operations for scientific computing:

```rust
use simdly::simd::SimdMath;

fn advanced_math_operations() {
    // Power operations
    let base = vec![2.0, 3.0, 4.0, 5.0];
    let exponent = vec![2.0, 2.0, 2.0, 2.0];
    let powers = base.pow(exponent);  // [4.0, 9.0, 16.0, 25.0]
    
    // 2D Euclidean distance
    let x = vec![3.0, 5.0, 8.0, 7.0];
    let y = vec![4.0, 12.0, 15.0, 24.0];
    let distances_2d = x.hypot(&y);   // [5.0, 13.0, 17.0, 25.0]
    
    // 3D Euclidean distance
    let z = vec![0.0, 5.0, 20.0, 0.0];
    let distances_3d = x.hypot3(&y, &z);
    
    // 4D Euclidean distance
    let w = vec![1.0, 0.0, 0.0, 7.0];
    let distances_4d = x.hypot4(&y, &z, &w);
    
    println!("2D distances: {:?}", distances_2d);
    println!("3D distances: {:?}", distances_3d);
    println!("4D distances: {:?}", distances_4d);
}
```

## Performance Optimization Patterns

### Algorithm Selection Strategy

```rust
use simdly::simd::SimdMath;
use simdly::SimdAdd;

// Smart algorithm selection based on data size
fn optimized_processing(data: Vec<f32>) -> Vec<f32> {
    match data.len() {
        // Small arrays: scalar operations are faster
        0..128 => data.into_iter().map(|x| x.cos()).collect(),
        
        // Medium arrays: SIMD operations are optimal
        128..262_144 => data.cos(),
        
        // Large arrays: parallel SIMD for maximum performance
        _ => data.par_cos(),
    }
}

// Addition with smart algorithm selection
fn smart_vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    match a.len() {
        0..128 => a.scalar_add(&b),
        128..262_144 => a.simd_add(&b),
        _ => a.par_simd_add(&b),
    }
}
```

### Parallel Processing for Large Datasets

```rust
use simdly::simd::SimdMath;

fn parallel_processing_example(data: Vec<f32>) -> Vec<f32> {
    // Parallel methods automatically handle PARALLEL_SIMD_THRESHOLD
    // No need to manually check array size!
    data.par_cos()  // Automatically uses parallel for arrays ≥ 262,144 elements
                    // Automatically uses single-threaded SIMD for smaller arrays
}

// Complex mathematical pipeline with parallel processing
fn parallel_math_pipeline(data: Vec<f32>) -> Vec<f32> {
    let threshold = 262_144;
    
    let step1 = if data.len() >= threshold {
        data.par_sin()       // Parallel SIMD sine
    } else {
        data.sin()           // Regular SIMD sine
    };
    
    let step2 = if step1.len() >= threshold {
        step1.par_abs()      // Parallel SIMD absolute value
    } else {
        step1.abs()          // Regular SIMD absolute value
    };
    
    if step2.len() >= threshold {
        step2.par_sqrt()     // Parallel SIMD square root
    } else {
        step2.sqrt()         // Regular SIMD square root
    }
}
```

### Operation Chaining

```rust
use simdly::simd::SimdMath;

// Efficient operation chaining
struct MathProcessor;

impl MathProcessor {
    // Chain multiple operations efficiently
    fn complex_transform(data: Vec<f32>) -> Vec<f32> {
        data.sin()          // Step 1: SIMD sine
            .abs()          // Step 2: SIMD absolute value
            .sqrt()         // Step 3: SIMD square root
    }
    
    // Conditional processing without branching
    fn conditional_transform(data: Vec<f32>, use_cosine: bool) -> Vec<f32> {
        if use_cosine {
            data.cos().exp()   // Cosine then exponential
        } else {
            data.sin().ln()    // Sine then natural log
        }
    }
    
    // Advanced mathematical pipeline
    fn scientific_pipeline(x: Vec<f32>, y: Vec<f32>) -> Vec<f32> {
        // Compute 2D distance, then apply mathematical transformations
        x.hypot(&y)         // 2D Euclidean distance
         .ln()              // Natural logarithm
         .abs()             // Absolute value
         .sqrt()            // Square root
    }
}
```

## Error Handling and Safety

### Safe High-Level Operations

simdly's high-level operations are safe and handle edge cases automatically:

```rust
use simdly::simd::SimdMath;
use simdly::SimdAdd;

pub fn safe_mathematical_operations(data: Vec<f32>) -> Vec<f32> {
    // All operations are safe and handle edge cases automatically
    data.cos()  // Automatically handles any array size
}

pub fn safe_vector_addition(a: Vec<f32>, b: Vec<f32>) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }
    
    // Choose the optimal algorithm automatically
    let result = match a.len() {
        0..128 => a.scalar_add(&b),
        128..262_144 => a.simd_add(&b),
        _ => a.par_simd_add(&b),
    };
    
    Ok(result)
}

// Complex operations with error handling
pub fn safe_complex_math(data: Vec<f32>) -> Result<Vec<f32>, &'static str> {
    if data.is_empty() {
        return Err("Input data cannot be empty");
    }
    
    // Chain operations safely
    let result = data
        .abs()      // Ensure positive values for sqrt
        .sqrt()     // Square root
        .cos();     // Cosine
    
    Ok(result)
}
```

## Best Practices

### 1. Use High-Level Mathematical Operations
Prefer `SimdMath` trait operations over low-level SIMD operations:

```rust
use simdly::simd::SimdMath;

// Good - high-level, safe, automatic optimization
let result = data.cos();

// Also good - algorithm selection
let result = match data.len() {
    0..128 => data.iter().map(|x| x.cos()).collect(),  // Scalar
    128..262_144 => data.cos(),                        // SIMD
    _ => data.par_cos(),                               // Parallel SIMD
};
```

### 2. Choose Algorithms Based on Data Size
Use the appropriate algorithm for your dataset size:

```rust
use simdly::SimdAdd;
use simdly::simd::SimdMath;

// Best - automatic algorithm selection
fn smart_processing_auto(data: Vec<f32>) -> Vec<f32> {
    data.par_cos()  // Automatically handles all thresholds!
}

// Manual algorithm selection (if you need explicit control)
fn smart_processing_manual(data: Vec<f32>) -> Vec<f32> {
    if data.len() >= 262_144 {
        data.par_cos()    // Parallel SIMD for large data
    } else if data.len() >= 128 {
        data.cos()        // Regular SIMD for medium data
    } else {
        data.into_iter().map(|x| x.cos()).collect()  // Scalar for small data
    }
}
```

### 3. Chain Operations Efficiently
Chain multiple mathematical operations for better performance:

```rust
use simdly::simd::SimdMath;

// Good - efficient operation chaining
let result = data
    .sin()      // SIMD sine
    .abs()      // SIMD absolute value
    .sqrt();    // SIMD square root

// Less efficient - separate operations
let step1 = data.sin();
let step2 = step1.abs();
let step3 = step2.sqrt();
```

### 4. Use Parallel Operations for Large Datasets
Leverage parallel SIMD for maximum performance on large data:

```rust
use simdly::simd::SimdMath;

// Best - automatic selection (recommended)
let result = data.par_cos();  // Automatically handles all thresholds!

// Manual selection (if you want explicit control)
if data.len() >= 262_144 {
    let result = data.par_cos();  // Uses multiple CPU cores
} else {
    let result = data.cos();      // Uses single-threaded SIMD
}

// Also manual - pattern matching
let result = match data.len() {
    262_144.. => data.par_cos(),
    128.. => data.cos(),
    _ => data.into_iter().map(|x| x.cos()).collect(),
};
```

## Next Steps

- Check out [Performance Tips](/guides/performance/) for more optimization strategies
- See the [API Reference](/reference/) for complete documentation of all available operations