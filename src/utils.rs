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

use std::alloc::{alloc, alloc_zeroed, handle_alloc_error, Layout};

/// Allocates an uninitialized f32 vector with specified alignment.
///
/// This function creates a vector of f32 values with custom memory alignment,
/// which may be required for certain SIMD operations or when interfacing with
/// external libraries that have specific alignment requirements.
///
/// # Performance Impact
///
/// **Warning**: This function introduces allocation overhead compared to standard
/// `Vec::with_capacity()`. Benchmarking has shown that for most ARM NEON operations,
/// standard Rust vectors provide better performance due to:
///
/// - Optimized memory management in the Rust allocator
/// - Reduced complexity in allocation logic
/// - Better integration with existing memory layout optimizations
/// - Elimination of custom alignment overhead
///
/// # Safety Considerations
///
/// The returned vector contains **uninitialized memory**. All elements must be
/// written to before reading to avoid undefined behavior. This is similar to
/// using `Vec::with_capacity()` followed by `set_len()`, but with custom alignment.
///
/// # Memory Layout
///
/// - **Alignment**: Memory is aligned to the specified `align` parameter
/// - **Size**: Allocated size is `len * sizeof(f32)` bytes
/// - **Capacity**: Vector capacity equals the requested length
/// - **Length**: Vector length is set to the requested length (unsafe)
///
/// # Arguments
///
/// * `len` - Number of f32 elements to allocate
/// * `align` - Memory alignment requirement in bytes (must be power of 2)
///
/// # Returns
///
/// A `Vec<f32>` with the specified length and alignment, containing uninitialized data.
///
/// # Panics
///
/// - Panics if `align` is not a power of 2
/// - Panics if the requested layout is invalid (size overflow)
/// - Calls `handle_alloc_error` if memory allocation fails
///
///
/// # Alternative (Recommended)
///
/// For most use cases, prefer the standard approach:
/// ```rust, ignore
/// let mut vec = Vec::with_capacity(len);
/// unsafe { vec.set_len(len) };
/// // Initialize elements as needed
/// ```
#[allow(dead_code)]
#[inline(always)]
pub fn alloc_uninit_f32_vec(len: usize, align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    let ptr = unsafe { alloc(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    unsafe { Vec::from_raw_parts(ptr, len, len) }
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
/// - Calls `handle_alloc_error` if memory allocation fails
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
#[allow(dead_code)]
#[inline(always)]
pub fn alloc_zeroed_f32_vec(len: usize, align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
