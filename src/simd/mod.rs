//! SIMD (Single Instruction, Multiple Data) operations module.
//!
//! This module provides traits and implementations for high-performance vectorized
//! operations using various CPU instruction sets. It includes abstractions for
//! memory alignment checking, data loading, and data storing operations.
//!
//! # Architecture Support
//!
//! The module conditionally compiles different implementations based on available
//! CPU features:
//!
//! - **AVX2**: 256-bit vectors for x86/x86_64 processors (when compiled with `avx2` feature)
//! - **SSE**: 128-bit vectors for x86/x86_64 processors (future implementation)
//! - **NEON**: 128-bit vectors for ARM/AArch64 processors (future implementation)
//!
//! # Usage
//!
//! The traits in this module provide a consistent interface across different SIMD
//! implementations, allowing code to be written generically and automatically use
//! the best available instruction set.

#[cfg(avx2)]
pub mod avx2;

/// Trait for checking memory alignment requirements.
///
/// Different SIMD instruction sets have different alignment requirements for
/// optimal performance. This trait provides a uniform way to check if a pointer
/// meets the alignment requirements for a specific SIMD implementation.
///
/// # Type Parameters
///
/// * `T` - The element type being checked for alignment
///
/// # Examples
///
/// ```rust
/// # use simdly::simd::{Alignment};
/// # struct MySimdType;
/// # impl Alignment<f32> for MySimdType {
/// #     fn is_aligned(ptr: *const f32) -> bool { true }
/// # }
/// let data = [1.0f32, 2.0, 3.0, 4.0];
/// let is_aligned = MySimdType::is_aligned(data.as_ptr());
/// ```
pub trait Alignment<T> {
    /// Checks if a pointer meets the alignment requirements.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to check for proper alignment
    ///
    /// # Returns
    ///
    /// `true` if the pointer is properly aligned, `false` otherwise
    fn is_aligned(ptr: *const T) -> bool;
}

/// Trait for loading data from memory into SIMD vectors.
///
/// This trait provides various methods for efficiently loading data from memory
/// into SIMD registers, with support for both aligned and unaligned memory access,
/// as well as partial loads for data that doesn't fill a complete vector.
///
/// # Type Parameters
///
/// * `T` - The element type being loaded
///
/// # Performance Considerations
///
/// - Aligned loads are generally faster than unaligned loads
/// - Partial loads use masking and may have additional overhead
/// - The `from_slice` method automatically chooses the best loading strategy
pub trait SimdLoad<T> {
    /// The output type returned by load operations
    type Output;

    /// High-level interface to load data from a slice.
    ///
    /// Automatically handles partial loads and chooses the most appropriate
    /// loading method based on data size and alignment.
    ///
    /// # Arguments
    ///
    /// * `slice` - Input slice containing data to load
    ///
    /// # Returns
    ///
    /// A SIMD vector containing the loaded data
    fn from_slice(slice: &[T]) -> Self::Output;

    /// Loads a complete vector from memory.
    ///
    /// Automatically detects alignment and uses the most efficient load operation.
    /// The size parameter must match the vector's lane count.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data to load
    /// * `size` - Number of elements to load (must equal vector lane count)
    ///
    /// # Safety
    ///
    /// Pointer must be valid and point to at least `size` elements.
    fn load(ptr: *const T, size: usize) -> Self::Output;

    /// Loads data from aligned memory.
    ///
    /// Provides optimal performance when data is properly aligned.
    /// Use `Alignment::is_aligned()` to verify alignment before calling.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Aligned pointer to data
    ///
    /// # Safety
    ///
    /// Pointer must be properly aligned and point to sufficient data.
    fn load_aligned(ptr: *const T) -> Self::Output;

    /// Loads data from unaligned memory.
    ///
    /// Works with any memory alignment but may be slower than aligned loads.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to sufficient valid data.
    fn load_unaligned(ptr: *const T) -> Self::Output;

    /// Loads fewer elements than the vector's full capacity.
    ///
    /// Uses masking operations to safely load partial data without reading
    /// beyond valid memory boundaries.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data
    /// * `size` - Number of elements to load (less than vector capacity)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least `size` valid elements.
    fn load_partial(ptr: *const T, size: usize) -> Self::Output;
}

/// Trait for storing SIMD vector data to memory.
///
/// This trait provides various methods for efficiently storing SIMD vector data
/// back to memory, with support for aligned/unaligned access, streaming stores,
/// and partial stores for vectors that don't completely fill the destination.
///
/// # Type Parameters
///
/// * `T` - The element type being stored
///
/// # Performance Considerations
///
/// - Aligned stores are generally faster than unaligned stores
/// - Streaming stores bypass cache and are optimal for large datasets
/// - Partial stores use masking and may have additional overhead
pub trait SimdStore<T> {
    /// The output type for store operations
    type Output;

    /// Stores vector data with automatic alignment detection.
    ///
    /// Automatically chooses between aligned and unaligned store based on
    /// the destination pointer's alignment.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must point to sufficient writable memory.
    fn store_at(&self, ptr: *const T);

    /// Non-temporal store that bypasses cache.
    ///
    /// Optimal for large datasets where data won't be accessed again soon.
    /// Prevents cache pollution during bulk memory operations.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must point to sufficient writable memory.
    fn stream_at(&self, ptr: *mut T);

    /// Stores data to aligned memory.
    ///
    /// Provides optimal performance when destination is properly aligned.
    /// Use `Alignment::is_aligned()` to verify alignment before calling.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Aligned mutable pointer to destination
    ///
    /// # Safety
    ///
    /// Pointer must be properly aligned and point to sufficient writable memory.
    fn store_aligned_at(&self, ptr: *mut T);

    /// Stores data to unaligned memory.
    ///
    /// Works with any memory alignment but may be slower than aligned stores.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to sufficient writable memory.
    fn store_unaligned_at(&self, ptr: *mut T);

    /// Stores only the valid elements using masked operations.
    ///
    /// Uses masking to safely store partial vector data without writing
    /// beyond the intended memory range.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must point to sufficient writable memory for the valid elements.
    fn store_at_partial(&self, ptr: *mut T);
}

pub trait SimdMath<T> {
    /// The output type returned by mathematical operations
    type Output;

    fn abs(&self) -> Self::Output;

    fn acos(&self) -> Self::Output;

    fn asin(&self) -> Self::Output;

    fn atan(&self) -> Self::Output;

    fn atan2(&self) -> Self::Output;

    fn cbrt(&self) -> Self::Output;

    fn floor(&self) -> Self::Output;

    fn exp(&self) -> Self::Output;

    fn ln(&self) -> Self::Output;

    fn hypot(&self) -> Self::Output;

    fn pow(&self) -> Self::Output;

    fn sin(&self) -> Self::Output;

    fn cos(&self) -> Self::Output;

    fn tan(&self) -> Self::Output;

    fn sqrt(&self) -> Self::Output;

    fn ceil(&self) -> Self::Output;

    fn hypot3(&self) -> Self::Output;

    fn hypot4(&self) -> Self::Output;
}
