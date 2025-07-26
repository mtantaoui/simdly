---
title: SIMD Operations
description: Learn about advanced SIMD operations and optimization techniques.
---

# SIMD Operations

This guide covers advanced SIMD operations available in simdly and how to use them effectively.

## Understanding SIMD

SIMD (Single Instruction, Multiple Data) allows you to perform the same operation on multiple data elements simultaneously. With AVX2, you can process 8 f32 values in parallel using 256-bit vectors.

## Core Operations

### Loading and Storing

simdly provides several loading and storing strategies optimized for different scenarios:

#### Aligned vs Unaligned Access

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{Alignment, SimdLoad, SimdStore};

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Check if pointer is aligned
let is_aligned = F32x8::is_aligned(data.as_ptr());

if is_aligned {
    // Use faster aligned operations
    let vec = unsafe { F32x8::load_aligned(data.as_ptr()) };
    // ... operations ...
    unsafe { vec.store_aligned_at(output.as_mut_ptr()) };
} else {
    // Use unaligned operations (slightly slower)
    let vec = unsafe { F32x8::load_unaligned(data.as_ptr()) };
    // ... operations ...
    unsafe { vec.store_unaligned_at(output.as_mut_ptr()) };
}
```

#### Streaming Stores

For large datasets where data won't be reused, use streaming stores to bypass cache:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::SimdStore;

let vec = F32x8::from_slice(&data);

// Stream to aligned memory (bypasses cache)
unsafe {
    vec.stream_at(aligned_output.as_mut_ptr());
}
```

### Partial Operations

Handle arrays that don't align perfectly with vector sizes:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

fn process_any_size_array(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    let mut i = 0;
    
    // Process full vectors (8 elements at a time)
    while i + 8 <= input.len() {
        let vec = F32x8::from_slice(&input[i..i + 8]);
        
        // Your SIMD operations here...
        
        unsafe {
            vec.store_unaligned_at(output[i..].as_mut_ptr());
        }
        i += 8;
    }
    
    // Handle remaining elements with partial operations
    if i < input.len() {
        let remaining = input.len() - i;
        let vec = unsafe { F32x8::load_partial(input[i..].as_ptr(), remaining) };
        
        // Your SIMD operations here...
        
        unsafe {
            vec.store_at_partial(output[i..].as_mut_ptr());
        }
    }
    
    output
}
```

## Performance Optimization Patterns

### Memory Access Patterns

#### Sequential Access (Optimal)
```rust
// Good: Sequential memory access
for chunk in data.chunks_exact(8) {
    let vec = F32x8::from_slice(chunk);
    // Process vector...
}
```

#### Strided Access (Less Optimal)
```rust
// Less optimal: Strided access requires gathering
// Consider restructuring data layout when possible
```

### Cache-Friendly Processing

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

fn cache_friendly_processing(data: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    
    // Process in cache-friendly chunks
    const CACHE_LINE_SIZE: usize = 64; // bytes
    const ELEMENTS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
    
    for chunk in data.chunks(ELEMENTS_PER_CACHE_LINE) {
        // Process each cache line with SIMD
        for simd_chunk in chunk.chunks(8) {
            let vec = F32x8::from_slice(simd_chunk);
            // Your operations...
        }
    }
    
    result
}
```

### Minimizing Memory Allocations

```rust
use simdly::simd::avx2::f32x8::F32x8;

// Pre-allocate buffers to avoid repeated allocations
struct SimdProcessor {
    temp_buffer: Vec<f32>,
}

impl SimdProcessor {
    fn new(max_size: usize) -> Self {
        Self {
            temp_buffer: vec![0.0; max_size],
        }
    }
    
    fn process(&mut self, input: &[f32]) -> &[f32] {
        self.temp_buffer.resize(input.len(), 0.0);
        
        // Process using pre-allocated buffer
        for (chunk_in, chunk_out) in input.chunks(8)
            .zip(self.temp_buffer.chunks_mut(8)) {
            
            let vec = F32x8::from_slice(chunk_in);
            // Your operations...
            
            unsafe {
                vec.store_unaligned_at(chunk_out.as_mut_ptr());
            }
        }
        
        &self.temp_buffer[..input.len()]
    }
}
```

## Error Handling and Safety

### Safe Wrappers

Create safe wrappers around unsafe SIMD operations:

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

pub fn safe_simd_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }
    
    let mut result = vec![0.0; a.len()];
    
    for i in (0..a.len()).step_by(8) {
        let chunk_size = std::cmp::min(8, a.len() - i);
        
        let vec_a = if chunk_size == 8 {
            F32x8::from_slice(&a[i..i + 8])
        } else {
            unsafe { F32x8::load_partial(a[i..].as_ptr(), chunk_size) }
        };
        
        let vec_b = if chunk_size == 8 {
            F32x8::from_slice(&b[i..i + 8])
        } else {
            unsafe { F32x8::load_partial(b[i..].as_ptr(), chunk_size) }
        };
        
        // Perform addition (would need actual SIMD add operation)
        // This is a placeholder - actual math operations would go here
        
        if chunk_size == 8 {
            unsafe {
                vec_a.store_unaligned_at(result[i..].as_mut_ptr());
            }
        } else {
            unsafe {
                vec_a.store_at_partial(result[i..].as_mut_ptr());
            }
        }
    }
    
    Ok(result)
}
```

## Best Practices

### 1. Prefer High-Level APIs
Use `from_slice()` over manual load operations when possible:

```rust
// Good
let vec = F32x8::from_slice(&data);

// Less good (manual)
let vec = unsafe { 
    if data.len() < 8 {
        F32x8::load_partial(data.as_ptr(), data.len())
    } else {
        F32x8::load(data.as_ptr(), 8)
    }
};
```

### 2. Handle Alignment Automatically
Let simdly handle alignment detection:

```rust
// Good - automatic alignment detection
let vec = unsafe { F32x8::load(ptr, 8) };

// Manual alignment check (unnecessary)
let vec = if F32x8::is_aligned(ptr) {
    unsafe { F32x8::load_aligned(ptr) }
} else {
    unsafe { F32x8::load_unaligned(ptr) }
};
```

### 3. Use Appropriate Store Operations
Choose the right store operation for your use case:

```rust
// For normal operations
unsafe { vec.store_unaligned_at(output.as_mut_ptr()) };

// For large datasets (bypasses cache)
unsafe { vec.stream_at(aligned_output.as_mut_ptr()) };

// For partial data
unsafe { vec.store_at_partial(output.as_mut_ptr()) };
```

### 4. Batch Process When Possible
Process multiple vectors in a loop for better performance:

```rust
// Good - batched processing
for chunk in data.chunks_exact(8) {
    let vec = F32x8::from_slice(chunk);
    // Process multiple operations on the same vector
    // ... operations ...
}

// Less efficient - processing one element at a time
for element in data {
    // Process individual elements
}
```

## Next Steps

- Check out [Performance Tips](/guides/performance/) for more optimization strategies
- See the [API Reference](/reference/) for complete documentation of all available operations