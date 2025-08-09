//! Memory allocation utilities for SIMD operations.
//!
//! This module provides specialized memory allocation functions for creating aligned
//! vectors required by some SIMD operations. While ARM NEON does not strictly require
//! aligned memory access, some optimizations and specific use cases may benefit from
//! or require aligned memory layouts.
//!
//! # Performance Considerations
//!
//! **Important Note**: Based on benchmarking analysis, the aligned allocation functions
//! in this module introduce measurable overhead compared to standard `Vec` allocation.
//! For most ARM NEON operations, the standard `Vec::with_capacity` approach is preferred
//! as it:
//!
//! - Eliminates custom allocation overhead
//! - Leverages Rust's optimized memory management
//! - Maintains compatibility with existing code
//! - Provides better performance for small to medium arrays
//!
//! # When to Use These Functions
//!
//! These allocation utilities should only be used when:
//! - Specific algorithms require guaranteed memory alignment
//! - Interfacing with C libraries that mandate aligned memory
//! - Implementing custom SIMD operations that benefit from alignment
//! - Benchmarking shows measurable performance improvements
//!
//! # Usage Recommendation
//!
//! For general SIMD operations in this codebase, prefer:
//! ```rust, ignore
//! let mut vec = Vec::with_capacity(size);
//! unsafe { vec.set_len(size) };
//! ```
//!
//! Over the aligned allocation functions provided here, unless specific alignment
//! requirements dictate otherwise.

use std::{
    alloc::{alloc, alloc_zeroed, handle_alloc_error, Layout},
    ptr::NonNull,
};

/// Allocates an uninitialized f32 vector with specified alignment.
///
/// **DANGER - UNINITIALIZED MEMORY**: This function allocates uninitialized memory
/// using custom alignment. The returned vector contains arbitrary data that must be
/// overwritten before reading to avoid undefined behavior.
///
/// This function creates a vector of f32 values with custom memory alignment using
/// direct memory allocation. The memory is **not initialized** and contains undefined
/// values that must be completely overwritten before use.
///
/// # Performance vs Safety Trade-off
///
/// **Performance**: Avoids zero-initialization overhead for cases where all elements
/// will be immediately overwritten by SIMD operations.
///
/// **Safety Risk**: The returned vector contains uninitialized memory. Reading from
/// it before writing will result in undefined behavior.
///
/// For safer alternatives, prefer:
/// ```rust, ignore
/// let vec = vec![0.0f32; len];  // Safe zero-initialized
/// let vec = alloc_zeroed_f32_vec(len, align);  // Safe aligned zero-initialized
/// ```
///
/// # Memory Layout
///
/// - **Alignment**: Memory is aligned to the specified `align` parameter
/// - **Size**: Allocated size is `len * sizeof(f32)` bytes
/// - **Capacity**: Vector capacity equals the requested length
/// - **Length**: Vector length is set to the requested length (but data is uninitialized)
/// - **Initialization**: **NONE - contains undefined data**
///
/// # Arguments
///
/// * `len` - Number of f32 elements to allocate
/// * `align` - Memory alignment requirement in bytes (must be power of 2)
///
/// # Returns
///
/// A `Vec<f32>` with the specified length and alignment containing **uninitialized data**.
///
/// # Panics
///
/// - Panics if `align` is not a power of 2
/// - Panics if the requested layout is invalid (size overflow)
/// - Panics with "Allocation failed" if memory allocation fails
///
/// # Safety Warning
///
/// This function is **unsafe** in that it returns uninitialized memory.
/// Callers must ensure all elements are written before reading to avoid undefined behavior.
///
/// # Usage Pattern
///
/// ```rust, ignore
/// let mut vec = alloc_uninit_f32_vec(1000, 32);
/// // MUST overwrite all elements before reading
/// for i in 0..vec.len() {
///     vec[i] = compute_value(i);  // Initialize each element
/// }
/// // Now safe to read from vec
/// ```
///
/// # Why This Function Exists
///
/// This function exists for performance-critical SIMD code where:
/// 1. All elements will be overwritten immediately by SIMD operations
/// 2. Zero-initialization overhead needs to be avoided
/// 3. Custom alignment is required for optimal SIMD performance
#[inline(always)]
pub(crate) fn alloc_uninit_f32_vec(len: usize, align: usize) -> Vec<f32> {
    assert!(align.is_power_of_two(), "Alignment must be a power of two");

    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    unsafe {
        let ptr = NonNull::new(alloc(layout) as *mut f32).expect("Allocation failed");

        // SAFETY: ptr came directly from Rust's allocator, capacity matches layout
        Vec::from_raw_parts(ptr.as_ptr(), len, len)
    }
}

/// Allocates a zero-initialized f32 vector with specified alignment.
///
/// This function creates a vector of f32 values with custom memory alignment,
/// where all elements are initialized to zero. This provides a safer alternative
/// to `alloc_uninit_f32_vec` when zero-initialization is acceptable.
///
/// # Performance Impact
///
/// **Double Warning**: This function has even higher overhead than the uninitialized
/// variant due to:
///
/// - Custom alignment allocation overhead
/// - Zero-initialization cost (memory must be written during allocation)
/// - Additional complexity compared to standard `Vec::new()` or `Vec::with_capacity()`
///
/// For most ARM NEON operations, prefer standard approaches:
/// ```rust, ignore
/// let vec = vec![0.0f32; len];  // For zero-initialized
/// ```
///
/// # Memory Layout
///
/// - **Alignment**: Memory is aligned to the specified `align` parameter
/// - **Size**: Allocated size is `len * sizeof(f32)` bytes
/// - **Capacity**: Vector capacity equals the requested length
/// - **Length**: Vector length is set to the requested length
/// - **Initialization**: All elements are set to 0.0
///
/// # Arguments
///
/// * `len` - Number of f32 elements to allocate
/// * `align` - Memory alignment requirement in bytes (must be power of 2)
///
/// # Returns
///
/// A `Vec<f32>` with the specified length and alignment, with all elements set to 0.0.
///
/// # Panics
///
/// - Panics if `align` is not a power of 2
/// - Panics if the requested layout is invalid (size overflow)
/// - Panics with "Allocation failed" if memory allocation fails
///
/// # Safety
///
/// Unlike `alloc_uninit_f32_vec`, this function returns a fully initialized vector
/// that is safe to read immediately without additional initialization.
///
/// # Alternative (Recommended)
///
/// For most use cases, prefer these standard approaches:
/// ```rust, ignore
/// // Simple zero-initialized vector
/// let vec = vec![0.0f32; len];
///
/// // Zero-initialized with explicit capacity
/// let mut vec = Vec::with_capacity(len);
/// vec.resize(len, 0.0);
/// ```
///
/// # Use Cases
///
/// This function should only be used when:
/// - Custom alignment is specifically required by external libraries
/// - Benchmarking demonstrates performance benefits for your specific use case
/// - Interfacing with C APIs that require aligned zero-initialized memory
#[inline(always)]
pub(crate) fn alloc_zeroed_f32_vec(len: usize, align: usize) -> Vec<f32> {
    assert!(align.is_power_of_two(), "Alignment must be a power of two");

    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    unsafe {
        let ptr = NonNull::new(alloc_zeroed(layout) as *mut f32).expect("Allocation failed");

        // SAFETY: ptr came directly from Rust's allocator, capacity matches layout
        Vec::from_raw_parts(ptr.as_ptr(), len, len)
    }
}
