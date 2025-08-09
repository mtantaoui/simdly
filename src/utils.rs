//! Memory allocation utilities for AVX2 SIMD operations.
//!
//! This module provides specialized memory allocation functions for creating 32-byte
//! aligned vectors required by AVX2 SIMD operations. These functions guarantee proper
//! alignment for optimal `__m256` performance while using safe Vec-based allocation
//! to prevent Windows heap corruption.
//!
//! # Key Features
//!
//! - **32-byte alignment**: Guaranteed alignment for AVX2 operations (`_mm256_*` intrinsics)
//! - **Windows-safe**: Uses Vec's standard allocator to prevent heap corruption
//! - **Two variants**: Uninitialized (faster) and zero-initialized (safer)
//! - **Memory efficient**: Uses `AlignedF32` wrapper type for proper alignment
//!
//! # When to Use These Functions
//!
//! These allocation utilities should be used when:
//! - AVX2 SIMD operations require guaranteed 32-byte alignment
//! - Performance-critical code needs to avoid initialization overhead (uninit version)
//! - Safe zero-initialized aligned memory is needed (zeroed version)
//! - Cross-platform compatibility is required (especially Windows)
//!
//! # Usage Recommendation
//!
//! **For most use cases, prefer standard Vec:**
//! ```rust, ignore
//! let vec = vec![0.0f32; len];  // Simple, safe, usually sufficient
//! ```
//!
//! **For AVX2-optimized code that needs guaranteed alignment:**
//! ```rust, ignore
//! let vec = alloc_zeroed_f32_vec(len, 32);    // Safe, aligned, zero-initialized
//! let vec = alloc_uninit_f32_vec(len, 32);    // Fast, aligned, must initialize manually
//! ```

use std::mem;

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct AlignedF32(f32);

/// Allocates an uninitialized f32 vector with 32-byte alignment.
///
/// **DANGER - UNINITIALIZED MEMORY**: This function allocates uninitialized memory
/// with 32-byte alignment for AVX2 operations. The returned vector contains arbitrary
/// data that must be overwritten before reading to avoid undefined behavior.
///
/// This function creates a vector of f32 values using Vec's standard allocator with
/// guaranteed 32-byte alignment via an aligned wrapper type. The memory is **not
/// initialized** and contains undefined values that must be completely overwritten before use.
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
/// - **Alignment**: Memory is guaranteed to be 32-byte aligned (AVX2 requirement)
/// - **Size**: Effective size is `len * sizeof(f32)` bytes  
/// - **Capacity**: Vector capacity may be larger than requested due to alignment requirements
/// - **Length**: Vector length is set to the requested length (but data is uninitialized)
/// - **Initialization**: **NONE - contains undefined data**
/// - **Implementation**: Uses `Vec<AlignedF32>` internally, converted to `Vec<f32>`
///
/// # Arguments
///
/// * `len` - Number of f32 elements to allocate
/// * `align` - **IGNORED** - Function always provides 32-byte alignment regardless of this parameter
///
/// # Returns
///
/// A `Vec<f32>` with the specified length and 32-byte alignment containing **uninitialized data**.
///
/// # Panics
///
/// - May panic if `Vec::with_capacity()` fails (out of memory)
/// - Generally very safe due to using standard Vec allocation
///
/// # Safety Warning
///
/// This function is **unsafe** in that it returns uninitialized memory.
/// Callers must ensure all elements are written before reading to avoid undefined behavior.
///
/// # Usage Pattern
///
/// ```rust, ignore
/// let mut vec = alloc_uninit_f32_vec(1000, 32);  // align param ignored, always 32-byte
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
/// 3. 32-byte alignment is required for optimal AVX2 performance
/// 4. Safe allocation/deallocation using Vec's standard allocator is needed
#[inline(always)]
pub(crate) fn alloc_uninit_f32_vec(len: usize, _align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let mut vec: Vec<AlignedF32> = Vec::with_capacity(len);
    let ptr = vec.as_mut_ptr() as *mut f32;
    let capacity = vec.capacity();

    // Prevent the original vector from being dropped and deallocating the memory
    mem::forget(vec);

    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

/// Allocates a zero-initialized f32 vector with 32-byte alignment.
///
/// **SAFE ZERO-INITIALIZED MEMORY**: This function allocates zero-initialized memory
/// with 32-byte alignment for AVX2 operations. All elements are set to 0.0 and safe
/// to read immediately. This provides a safer alternative to `alloc_uninit_f32_vec`
/// when zero-initialization is acceptable.
///
/// This function creates a vector of f32 values using Vec's standard allocator with
/// guaranteed 32-byte alignment via an aligned wrapper type. All elements are
/// initialized to 0.0.
///
/// # Performance vs Safety Trade-off
///
/// **Safety**: All elements are zero-initialized and safe to read immediately.
///
/// **Performance**: Has initialization overhead compared to `alloc_uninit_f32_vec`,
/// but still uses efficient Vec allocation and avoids Windows heap corruption.
///
/// For different use cases, consider:
/// ```rust, ignore
/// let vec = vec![0.0f32; len];  // Simple, no alignment guarantee
/// let vec = alloc_uninit_f32_vec(len, 32);  // Faster, uninitialized
/// ```
///
/// # Memory Layout
///
/// - **Alignment**: Memory is guaranteed to be 32-byte aligned (AVX2 requirement)
/// - **Size**: Effective size is `len * sizeof(f32)` bytes
/// - **Capacity**: Vector capacity may be larger than requested due to alignment requirements
/// - **Length**: Vector length is set to the requested length
/// - **Initialization**: All elements are set to 0.0 (safe to read immediately)
/// - **Implementation**: Uses `Vec<AlignedF32>` internally, converted to `Vec<f32>`
///
/// # Arguments
///
/// * `len` - Number of f32 elements to allocate
/// * `align` - **IGNORED** - Function always provides 32-byte alignment regardless of this parameter
///
/// # Returns
///
/// A `Vec<f32>` with the specified length and 32-byte alignment, all elements set to 0.0.
///
/// # Panics
///
/// - May panic if `Vec::with_capacity()` fails (out of memory)
/// - Generally very safe due to using standard Vec allocation
///
/// # Safety
///
/// This function is completely safe - all elements are zero-initialized and can be
/// read immediately without additional initialization. No risk of undefined behavior.
///
/// # Usage Pattern
///
/// ```rust, ignore
/// let vec = alloc_zeroed_f32_vec(1000, 32);  // align param ignored, always 32-byte
/// // Safe to read immediately - all elements are 0.0
/// println!("First element: {}", vec[0]);  // Safe: prints 0.0
///
/// // Can also write to elements as needed
/// for i in 0..vec.len() {
///     vec[i] = compute_value(i);  // Optional: overwrite with computed values
/// }
/// ```
///
/// # Why This Function Exists
///
/// This function exists for SIMD code that needs:
/// 1. Zero-initialized memory that's safe to read immediately
/// 2. 32-byte alignment for optimal AVX2 performance
/// 3. Safe allocation/deallocation using Vec's standard allocator
/// 4. No risk of Windows heap corruption
#[inline(always)]
pub(crate) fn alloc_zeroed_f32_vec(len: usize, _align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let mut vec: Vec<AlignedF32> = vec![AlignedF32(0.0); len];
    let ptr = vec.as_mut_ptr() as *mut f32;
    let capacity = vec.capacity();

    // Prevent the original vector from being dropped and deallocating the memory
    mem::forget(vec);

    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}
