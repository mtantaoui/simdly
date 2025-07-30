---
title: Quick Start
description: Get up and running with Simdly cross-platform SIMD in minutes.
---

# Quick Start Guide

This guide will get you up and running with Simdly cross-platform SIMD in just a few minutes. Simdly automatically uses AVX2 on x86/x86_64 and NEON on ARM architectures.

## Basic Usage

Here's a simple cross-platform example:

```rust
use simdly::f32x8;

fn main() {
    // Create some data
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    // Load into SIMD vector (works on both AVX2 and NEON)
    let vec = f32x8::load_unaligned(&data);
    
    // Perform vector operations
    let doubled = vec.mul(f32x8::splat(2.0));
    
    println!("Processing {} lanes using universal SIMD API", f32x8::lanes());
}
```

## Loading Data

### Full Vector Loading

```rust
use simdly::f32x8;

// Platform-adaptive loading (8 elements on AVX2, 4 on NEON)
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = f32x8::load_unaligned(&data);
assert_eq!(f32x8::lanes(), vec.len()); // Check platform lane count
```

### Partial Loading

```rust
use simdly::f32x8;

let data = [1.0, 2.0, 3.0]; // Fewer elements than vector lanes
let vec = f32x8::load_partial(&data, data.len());
// Automatically handles partial loads on both platforms
```

### Aligned vs Unaligned Loading

```rust
use simdly::f32x8;

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Unaligned (works with any memory)
let vec1 = f32x8::load_unaligned(&data);

// Aligned (requires proper alignment - 32 bytes on x86, 16 on ARM)
if f32x8::is_aligned(data.as_ptr()) {
    let vec2 = f32x8::load_aligned(&data);
}
```

## Storing Data

### Basic Storing

```rust
use simdly::f32x8;

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = f32x8::load_unaligned(&data);

// Process the vector
let doubled = vec.mul(f32x8::splat(2.0));

// Store results (platform-adaptive)
let mut output = [0.0f32; 8];
doubled.store_unaligned(&mut output);
```

### Partial Storing

```rust
use simdly::f32x8;

let data = [1.0, 2.0, 3.0]; // Only 3 elements
let vec = f32x8::load_partial(&data, data.len());
let doubled = vec.mul(f32x8::splat(2.0));

let mut output = [0.0f32; 8];
doubled.store_partial(&mut output, data.len());

// Only first 3 elements are written
assert_eq!(&output[..3], &[2.0, 4.0, 6.0]);
assert_eq!(&output[3..], &[0.0; 5]); // Rest remain zero
```

## Memory Alignment

For optimal performance, consider using platform-appropriate aligned memory:

```rust
use simdly::f32x8;
use std::alloc::{alloc, dealloc, Layout};

// Platform-appropriate alignment (32 bytes on x86, 16 on ARM)
fn get_optimal_alignment() -> usize {
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        32  // AVX2 alignment
    } else {
        16  // NEON alignment
    }
}

let alignment = get_optimal_alignment();
let layout = Layout::from_size_align(8 * std::mem::size_of::<f32>(), alignment).unwrap();
let aligned_ptr = unsafe { alloc(layout) as *mut f32 };

// Check alignment
assert!(f32x8::is_aligned(aligned_ptr));

// Use aligned operations for better performance
let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
unsafe {
    std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
    let slice = std::slice::from_raw_parts(aligned_ptr, f32x8::lanes());
    let vec = f32x8::load_aligned(slice);
    vec.store_aligned(slice);
}

// Clean up
unsafe { dealloc(aligned_ptr as *mut u8, layout) };
```

## Performance Tips

1. **Zero Configuration**: Simdly automatically uses optimal SIMD features thanks to its built-in `.cargo/config.toml` with `target-cpu=native`. No setup required!

2. **Use Platform-Appropriate Alignment**: 32-byte on x86, 16-byte on ARM.

3. **Adaptive Batch Processing**: Use `f32x8::lanes()` to get platform vector size.

4. **Release Mode**: Always benchmark in release mode with optimizations enabled.

5. **Cross-Platform Code**: Use the universal API for code that runs optimally on both architectures.

6. **Automatic Optimization**: Just `cargo build --release` and Simdly automatically uses the best instructions for your CPU.

## Next Steps

- Learn about [Cross-Platform SIMD Operations](/guides/simd-operations/) for more advanced usage
- Check out [Performance Tips](/guides/performance/) for optimization strategies across architectures
- Explore the [API Reference](/reference/) for detailed universal API documentation

## Common Patterns

### Cross-Platform Array Processing

```rust
use simdly::f32x8;

fn process_array(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    let lane_count = f32x8::lanes();
    let chunks = input.len() / lane_count;
    let remainder = input.len() % lane_count;
    
    // Process full chunks (platform-adaptive)
    for i in 0..chunks {
        let start = i * lane_count;
        let vec = f32x8::load_unaligned(&input[start..]);
        
        // Your SIMD operations here (example: multiply by 2)
        let result = vec.mul(f32x8::splat(2.0));
        
        result.store_unaligned(&mut output[start..]);
    }
    
    // Handle remainder with partial operations
    if remainder > 0 {
        let start = chunks * lane_count;
        let vec = f32x8::load_partial(&input[start..], remainder);
        
        // Your SIMD operations here
        let result = vec.mul(f32x8::splat(2.0));
        
        result.store_partial(&mut output[start..], remainder);
    }
}
```

This pattern efficiently processes arrays of any size using cross-platform SIMD operations that automatically adapt to AVX2 (8 lanes) or NEON (4 lanes).