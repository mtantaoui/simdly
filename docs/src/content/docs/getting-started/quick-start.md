---
title: Quick Start
description: Get up and running with simdly in minutes.
---

# Quick Start Guide

This guide will get you up and running with simdly in just a few minutes.

## Basic Usage

Here's a simple example to get you started:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdLoad;

fn main() {
    // Create some data
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    // Load into SIMD vector
    let vec = F32x8::from_slice(&data);
    
    println!("Loaded {} elements into SIMD vector", vec.size);
}
```

## Loading Data

### Full Vector Loading (8 elements)

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdLoad;

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = F32x8::from_slice(&data);
assert_eq!(vec.size, 8);
```

### Partial Loading (< 8 elements)

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdLoad;

let data = [1.0, 2.0, 3.0]; // Only 3 elements
let vec = F32x8::from_slice(&data);
assert_eq!(vec.size, 3); // Only 3 valid elements
```

### Oversized Arrays

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdLoad;

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let vec = F32x8::from_slice(&data); // Takes first 8 elements
assert_eq!(vec.size, 8);
```

## Storing Data

### Basic Storing

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = F32x8::from_slice(&data);

let mut output = [0.0f32; 8];
unsafe {
    vec.store_unaligned_at(output.as_mut_ptr());
}

assert_eq!(output, data);
```

### Partial Storing

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

let data = [1.0, 2.0, 3.0]; // Only 3 elements
let vec = F32x8::from_slice(&data);

let mut output = [0.0f32; 8];
unsafe {
    vec.store_at_partial(output.as_mut_ptr());
}

// Only first 3 elements are written
assert_eq!(&output[..3], &data);
assert_eq!(&output[3..], &[0.0; 5]); // Rest remain zero
```

## Memory Alignment

For optimal performance, consider using aligned memory:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{Alignment, SimdLoad, SimdStore};
use std::alloc::{alloc, dealloc, Layout};

// Allocate 32-byte aligned memory
let layout = Layout::from_size_align(8 * std::mem::size_of::<f32>(), 32).unwrap();
let aligned_ptr = unsafe { alloc(layout) as *mut f32 };

// Check alignment
assert!(F32x8::is_aligned(aligned_ptr));

// Use aligned operations for better performance
let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
unsafe {
    std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
    let vec = F32x8::load_aligned(aligned_ptr);
    vec.store_aligned_at(aligned_ptr);
}

// Clean up
unsafe { dealloc(aligned_ptr as *mut u8, layout) };
```

## Performance Tips

1. **Enable AVX2**: Always compile with AVX2 enabled:
   ```bash
   RUSTFLAGS="-C target-feature=+avx2" cargo build --release
   ```

2. **Use Aligned Memory**: When possible, use 32-byte aligned memory for optimal performance.

3. **Batch Processing**: Process data in chunks that match SIMD vector sizes (8 elements for F32x8).

4. **Release Mode**: Always benchmark in release mode with optimizations enabled.

## Next Steps

- Learn about [SIMD Operations](/guides/simd-operations/) for more advanced usage
- Check out [Performance Tips](/guides/performance/) for optimization strategies
- Explore the [API Reference](/reference/) for detailed documentation

## Common Patterns

### Processing Arrays

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

fn process_array(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    let chunks = input.len() / 8;
    let remainder = input.len() % 8;
    
    // Process full chunks
    for i in 0..chunks {
        let start = i * 8;
        let vec = F32x8::from_slice(&input[start..start + 8]);
        
        // Your SIMD operations here...
        
        unsafe {
            vec.store_unaligned_at(output[start..].as_mut_ptr());
        }
    }
    
    // Handle remainder
    if remainder > 0 {
        let start = chunks * 8;
        let vec = F32x8::from_slice(&input[start..]);
        
        // Your SIMD operations here...
        
        unsafe {
            vec.store_at_partial(output[start..].as_mut_ptr());
        }
    }
}
```

This pattern efficiently processes arrays of any size using SIMD operations.