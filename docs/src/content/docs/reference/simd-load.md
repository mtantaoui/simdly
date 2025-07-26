---
title: SimdLoad Trait
description: Trait for loading data from memory into SIMD vectors.
---

# SimdLoad Trait

The `SimdLoad` trait provides various methods for efficiently loading data from memory into SIMD registers, with support for both aligned and unaligned memory access, as well as partial loads for data that doesn't fill a complete vector.

## Definition

```rust
pub trait SimdLoad<T> {
    type Output;

    fn from_slice(slice: &[T]) -> Self::Output;
    unsafe fn load(ptr: *const T, size: usize) -> Self::Output;
    unsafe fn load_aligned(ptr: *const T) -> Self::Output;
    unsafe fn load_unaligned(ptr: *const T) -> Self::Output;
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self::Output;
}
```

## Associated Types

### `Output`

The output type returned by load operations. For `F32x8`, this is `F32x8` itself.

## Methods

### `from_slice(slice: &[T]) -> Self::Output`

High-level interface to load data from a slice. This is the recommended method for most use cases.

**Behavior:**
- Automatically handles partial loads for slices smaller than the vector capacity
- Chooses the most appropriate loading method based on data size and alignment
- For slices >= vector capacity: Loads the first N elements using full loading
- For slices < vector capacity: Uses masked partial loading

**Example:**
```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

// Full vector (8 elements)
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let vec = F32x8::from_slice(&data);
assert_eq!(vec.size, 8);

// Partial vector (3 elements)
let partial = [1.0, 2.0, 3.0];
let vec = F32x8::from_slice(&partial);
assert_eq!(vec.size, 3);

// Oversized array (takes first 8)
let large = [1.0; 16];
let vec = F32x8::from_slice(&large);
assert_eq!(vec.size, 8);
```

**Panics:**
- In debug builds: panics if the slice is empty

### `unsafe fn load(ptr: *const T, size: usize) -> Self::Output`

Loads a complete vector from memory with automatic alignment detection.

**Parameters:**
- `ptr`: Pointer to data to load
- `size`: Number of elements to load (must equal vector lane count)

**Behavior:**
- Automatically detects alignment and uses the most efficient load operation
- Uses `load_aligned()` if pointer is properly aligned
- Uses `load_unaligned()` if pointer is not aligned

**Example:**
```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

let data = [1.0f32; 8];
let vec = unsafe { F32x8::load(data.as_ptr(), 8) };
assert_eq!(vec.size, 8);
```

**Safety:**
- Pointer must not be null and must point to at least `size` valid elements

**Panics:**
- In debug builds: panics if `size` != vector lane count or if pointer is null

### `unsafe fn load_aligned(ptr: *const T) -> Self::Output`

Loads data from aligned memory (fastest option).

**Parameters:**
- `ptr`: Aligned pointer to data

**Behavior:**
- Provides optimal performance when data is properly aligned
- Uses optimized aligned load instructions (e.g., `_mm256_load_ps` for AVX2)

**Example:**
```rust
use simdly::simd::{Alignment, SimdLoad, avx2::f32x8::F32x8};
use std::alloc::{alloc, Layout};

// Allocate aligned memory
let layout = Layout::from_size_align(8 * std::mem::size_of::<f32>(), 32).unwrap();
let aligned_ptr = unsafe { alloc(layout) as *mut f32 };

// Verify alignment
assert!(F32x8::is_aligned(aligned_ptr));

// Use aligned load
let vec = unsafe { F32x8::load_aligned(aligned_ptr) };
```

**Safety:**
- Pointer must be properly aligned (32-byte boundary for AVX2)
- Pointer must point to sufficient valid data
- Use `Alignment::is_aligned()` to verify alignment before calling

### `unsafe fn load_unaligned(ptr: *const T) -> Self::Output`

Loads data from unaligned memory.

**Parameters:**
- `ptr`: Pointer to data (no alignment requirement)

**Behavior:**
- Works with any memory alignment
- Slightly slower than aligned loads but more flexible
- Uses unaligned load instructions (e.g., `_mm256_loadu_ps` for AVX2)

**Example:**
```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let unaligned_ptr = unsafe { data.as_ptr().add(1) }; // Skip first element

let vec = unsafe { F32x8::load_unaligned(unaligned_ptr) };
// Loads [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

**Safety:**
- Pointer must point to sufficient valid data

### `unsafe fn load_partial(ptr: *const T, size: usize) -> Self::Output`

Loads fewer elements than the vector's full capacity using masked operations.

**Parameters:**
- `ptr`: Pointer to data
- `size`: Number of elements to load (must be less than vector capacity)

**Behavior:**
- Uses masking operations to safely load partial data
- Prevents reading beyond valid memory boundaries
- Unloaded lanes contain undefined values
- Sets the vector's `size` field to the actual number of loaded elements

**Example:**
```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

// Load only 3 elements
let data = [1.0, 2.0, 3.0];
let vec = unsafe { F32x8::load_partial(data.as_ptr(), 3) };
assert_eq!(vec.size, 3);

// Load 7 elements
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
let vec = unsafe { F32x8::load_partial(data.as_ptr(), 7) };
assert_eq!(vec.size, 7);
```

**Safety:**
- Pointer must not be null and must point to at least `size` valid elements

**Panics:**
- In debug builds: panics if `size` >= vector capacity or if pointer is null

## Performance Considerations

### Load Method Performance (from fastest to slowest)

1. **`load_aligned()`**: Fastest, requires 32-byte aligned memory
2. **`load_unaligned()`**: Slightly slower, works with any alignment
3. **`load_partial()`**: Additional overhead due to masking operations

### Recommendations

1. **Use `from_slice()` for most cases**: It automatically chooses the best strategy
2. **Prefer aligned memory when possible**: Use `load_aligned()` for maximum performance
3. **Batch partial loads**: Process multiple partial vectors together when possible

## Usage Patterns

### Processing Arrays of Any Size

```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

fn load_any_size_array(data: &[f32]) -> Vec<F32x8> {
    let mut vectors = Vec::new();
    let mut i = 0;
    
    // Load full vectors
    while i + 8 <= data.len() {
        let vec = unsafe { F32x8::load(data[i..].as_ptr(), 8) };
        vectors.push(vec);
        i += 8;
    }
    
    // Load remainder as partial vector
    if i < data.len() {
        let remaining = data.len() - i;
        let vec = unsafe { F32x8::load_partial(data[i..].as_ptr(), remaining) };
        vectors.push(vec);
    }
    
    vectors
}
```

### Alignment-Aware Loading

```rust
use simdly::simd::{Alignment, SimdLoad, avx2::f32x8::F32x8};

fn alignment_aware_load(data: &[f32]) -> F32x8 {
    debug_assert!(data.len() >= 8);
    
    let ptr = data.as_ptr();
    
    if F32x8::is_aligned(ptr) {
        unsafe { F32x8::load_aligned(ptr) }
    } else {
        unsafe { F32x8::load_unaligned(ptr) }
    }
}
```

### Safe Wrapper

```rust
use simdly::simd::{SimdLoad, avx2::f32x8::F32x8};

pub fn safe_load_exact(data: &[f32]) -> Result<F32x8, &'static str> {
    if data.len() < 8 {
        return Err("Data must contain at least 8 elements");
    }
    
    Ok(unsafe { F32x8::load(data.as_ptr(), 8) })
}

pub fn safe_load_partial(data: &[f32]) -> Result<F32x8, &'static str> {
    if data.is_empty() {
        return Err("Data cannot be empty");
    }
    
    if data.len() >= 8 {
        Ok(unsafe { F32x8::load(data.as_ptr(), 8) })
    } else {
        Ok(unsafe { F32x8::load_partial(data.as_ptr(), data.len()) })
    }
}
```

## Implementation Details

### AVX2 Implementation

For `F32x8`, the trait methods map to these AVX2 intrinsics:

- `load_aligned()` → `_mm256_load_ps()`
- `load_unaligned()` → `_mm256_loadu_ps()`
- `load_partial()` → `_mm256_maskload_ps()` with dynamically generated masks

### Masking for Partial Loads

Partial loads use bitmasks to control which elements are loaded:

```rust
// Example: Loading 3 elements uses this mask
let mask = _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0);
//                            ^   ^   ^  ^  ^  ^  ^  ^
//                            Load these | Skip these
```

## Related

- [SimdStore trait](/reference/simd-store/) - Storing operations
- [F32x8 struct](/reference/f32x8/) - AVX2 SIMD vector implementation
- [Alignment trait](/reference/alignment/) - Memory alignment checking