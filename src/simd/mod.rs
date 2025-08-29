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
//! - **NEON**: 128-bit vectors for ARM/AArch64 processors (when compiled with `neon` feature)
//! - **SSE**: 128-bit vectors for x86/x86_64 processors (future implementation)
//!
//! # Usage
//!
//! The traits in this module provide a consistent interface across different SIMD
//! implementations, allowing code to be written generically and automatically use
//! the best available instruction set.

#[cfg(avx2)]
pub mod avx2;

#[cfg(neon)]
pub mod neon;

pub mod slice;

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
/// # Performance Impact
///
/// Proper alignment can significantly improve SIMD performance:
/// - **Aligned loads/stores**: Can be 2-4x faster than unaligned operations
/// - **Memory bandwidth**: Better utilization of cache lines and memory buses
/// - **Instruction efficiency**: Some SIMD instructions require aligned data
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
/// if is_aligned {
///     // Use fast aligned operations
/// } else {
///     // Fall back to unaligned operations
/// }
/// ```
pub trait Alignment<T> {
    /// Checks if a pointer meets the alignment requirements.
    ///
    /// This method should be called before using aligned SIMD operations
    /// to ensure optimal performance and avoid potential undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to check for proper alignment
    ///
    /// # Returns
    ///
    /// `true` if the pointer is properly aligned for this SIMD type, `false` otherwise
    ///
    /// # Performance
    ///
    /// This function is typically very fast (single bitwise operation) and should
    /// be inlined by the compiler for zero-cost abstraction.
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
/// - The `From<&[T]>` trait implementation automatically chooses the best loading strategy
pub trait SimdLoad<T> {
    /// The output type returned by load operations
    type Output;

    /// High-level interface to load data from a slice using the `From` trait.
    ///
    /// Use `VectorType::from(&slice)` to create vectors from slices.
    /// This automatically handles partial loads and chooses the most appropriate
    /// loading method based on data size and alignment. This is the recommended
    /// approach for most use cases as it provides optimal performance automatically.
    ///
    /// # Performance
    ///
    /// The `From` trait implementation automatically selects the fastest loading strategy:
    /// - For aligned data: Uses fast aligned loads
    /// - For unaligned data: Uses slower but safe unaligned loads
    /// - For partial data: Uses masked loads to prevent buffer overruns
    ///
    /// Loads a complete vector from memory.
    ///
    /// Automatically detects alignment and uses the most efficient load operation.
    /// The size parameter must match the vector's lane count exactly.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data to load
    /// * `size` - Number of elements to load (must equal vector lane count)
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and non-null
    /// - Must point to at least `size` consecutive, initialized elements
    /// - Elements must remain valid for the duration of the load operation
    /// - Size must exactly match the vector's lane count
    ///
    /// # Panics
    ///
    /// May panic in debug builds if size doesn't match the expected lane count.
    unsafe fn load(ptr: *const T, size: usize) -> Self::Output;

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
    /// - Pointer must be properly aligned for this SIMD type
    /// - Must be valid and non-null
    /// - Must point to at least enough consecutive, initialized elements to fill the vector
    /// - Undefined behavior if alignment requirements are not met
    ///
    /// # Performance
    ///
    /// This is typically the fastest loading method available, as it can use
    /// the most efficient CPU instructions designed for aligned data access.
    unsafe fn load_aligned(ptr: *const T) -> Self::Output;

    /// Loads data from unaligned memory.
    ///
    /// Works with any memory alignment but may be slower than aligned loads.
    /// Use this when alignment cannot be guaranteed.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data (no alignment requirement)
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and non-null
    /// - Must point to at least enough consecutive, initialized elements to fill the vector
    /// - No alignment requirements, but data must be properly initialized
    ///
    /// # Performance
    ///
    /// Typically 10-20% slower than aligned loads, but still much faster than
    /// scalar operations. Modern CPUs have optimized unaligned access.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Output;

    /// Loads fewer elements than the vector's full capacity.
    ///
    /// Uses masking operations to safely load partial data without reading
    /// beyond valid memory boundaries. Remaining vector lanes will contain
    /// undefined values.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to data
    /// * `size` - Number of elements to load (less than vector capacity)
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and non-null
    /// - Must point to at least `size` consecutive, initialized elements
    /// - Size must be less than the vector's lane count
    ///
    /// # Performance
    ///
    /// Slightly slower than full loads due to masking overhead, but prevents
    /// buffer overruns and segmentation faults when working with partial data.
    ///
    /// # Panics
    ///
    /// May panic in debug builds if size is greater than or equal to the vector capacity.
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self::Output;
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
    /// the destination pointer's alignment. This is the recommended method
    /// for most use cases.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to destination memory
    ///
    /// # Safety
    ///
    /// - Pointer must be valid and non-null
    /// - Must point to sufficient writable memory for all vector elements
    /// - Memory must remain valid and writable during the store operation
    ///
    /// # Performance
    ///
    /// Automatically selects the fastest store method based on alignment,
    /// providing optimal performance without manual alignment checking.
    fn store_at(&self, ptr: *const T);

    /// Non-temporal store that bypasses cache.
    ///
    /// Optimal for large datasets where data won't be accessed again soon.
    /// Prevents cache pollution during bulk memory operations by writing
    /// directly to main memory.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// - Pointer must be valid, non-null, and properly aligned
    /// - Must point to sufficient writable memory for all vector elements
    /// - Memory must remain valid during the store operation
    ///
    /// # Performance
    ///
    /// Best for:
    /// - Large sequential writes that won't be read soon
    /// - Avoiding cache pollution in streaming algorithms
    /// - Write-once, read-never or read-much-later patterns
    ///
    /// Avoid for data that will be accessed again soon.
    unsafe fn stream_at(&self, ptr: *mut T);

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
    /// - Pointer must be properly aligned for this SIMD type
    /// - Must be valid, non-null, and writable
    /// - Must point to sufficient memory for all vector elements
    /// - Undefined behavior if alignment requirements are not met
    ///
    /// # Performance
    ///
    /// This is typically the fastest store method, using the most efficient
    /// CPU instructions designed for aligned memory access.
    unsafe fn store_aligned_at(&self, ptr: *mut T);

    /// Stores data to unaligned memory.
    ///
    /// Works with any memory alignment but may be slower than aligned stores.
    /// Use when alignment cannot be guaranteed.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination (no alignment requirement)
    ///
    /// # Safety
    ///
    /// - Pointer must be valid, non-null, and writable
    /// - Must point to sufficient memory for all vector elements
    /// - No alignment requirements
    ///
    /// # Performance
    ///
    /// Typically 10-20% slower than aligned stores, but still much faster
    /// than scalar operations. Modern CPUs handle unaligned access well.
    unsafe fn store_unaligned_at(&self, ptr: *mut T);

    /// Stores only the valid elements using masked operations.
    ///
    /// Uses masking to safely store partial vector data without writing
    /// beyond the intended memory range. Only stores elements up to the
    /// vector's valid size.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// - Pointer must be valid, non-null, and writable
    /// - Must point to sufficient memory for the vector's valid elements
    /// - Safe from buffer overruns due to masking
    ///
    /// # Performance
    ///
    /// Slightly slower than full stores due to masking overhead, but prevents
    /// writing beyond buffer boundaries when working with partial data.
    unsafe fn store_at_partial(&self, ptr: *mut T);
}

/// Trait for SIMD mathematical operations on f32 vectors.
///
/// This trait provides vectorized implementations of common mathematical functions
/// that operate on SIMD vectors. Each function processes multiple f32 values simultaneously
/// using optimized CPU instructions (AVX2, NEON, etc.) for maximum performance.
///
/// # Platform Support
///
/// - **x86/x86_64**: Uses AVX2 instructions with 8-element vectors (256-bit)
/// - **ARM/AArch64**: Uses NEON instructions with 4-element vectors (128-bit)
/// - **Fallback**: Scalar implementations for unsupported architectures
///
/// # Performance Characteristics
///
/// - **Vectorization**: All operations process multiple elements per instruction
/// - **Speedup**: Typically 2-13x faster than scalar operations (function-dependent)
/// - **Precision**: Maintains high accuracy comparable to scalar implementations
/// - **Special Values**: Proper handling of NaN, infinity, and edge cases according to IEEE 754
/// - **Domain Validation**: Input range checking for domain-restricted functions
///
/// # Usage Examples
///
/// ## High-Level Usage (Recommended)
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let angles = vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0];
/// let cosines = angles.cos(); // Automatic SIMD acceleration
/// let roots = angles.sqrt();  // Vectorized square root
/// ```
///
/// ## Platform-Specific Usage
/// ```rust
/// #[cfg(target_arch = "x86_64")]{
/// use simdly::simd::avx2::f32x8::F32x8;
/// use simdly::simd::SimdMath;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let vec = F32x8::from(data.as_slice());
/// let result = vec.sin(); // 8 parallel sine calculations
/// }
/// ```
///
/// # Implementation Notes
///
/// Implementations should:
/// - Use optimized SIMD instructions where available (AVX2 `_mm256_*`, NEON `v*q_f32`)
/// - Handle special values according to IEEE 754 standards
/// - Provide consistent accuracy across all vector lanes
/// - Maintain thread safety for concurrent usage
/// - Fall back to scalar operations when SIMD is unavailable
pub trait SimdMath {
    /// The output type returned by mathematical operations
    type Output;

    /// Computes the absolute value of each element.
    ///
    /// Returns |x| for each element x in the vector. This operation is typically very fast
    /// as it only requires clearing the sign bit using bitwise operations.
    ///
    /// # Performance
    /// - **Operation**: Sign bit manipulation (very fast)
    /// - **Typical speedup**: 4-8x over scalar
    /// - **Special cases**: -0.0 becomes +0.0, preserves NaN
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    /// let data = vec![-2.5, 1.0, -0.0, f32::INFINITY];
    /// let result = data.abs(); // [2.5, 1.0, 0.0, INFINITY]
    /// ```
    fn abs(&self) -> Self::Output;

    /// Computes the arccosine (inverse cosine) of each element.
    ///
    /// Returns the angle in radians whose cosine is the input value.
    /// Uses polynomial approximation for high performance while maintaining accuracy.
    ///
    /// # Domain & Range
    /// - **Input**: [-1, 1] (values outside this range return NaN)
    /// - **Output**: [0, π] radians
    /// - **Accuracy**: Typically within 1-2 ULP of standard library
    ///
    /// # Performance
    /// - **Implementation**: Polynomial approximation with range reduction
    /// - **Typical speedup**: 3-6x over scalar
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    /// let data = vec![1.0, 0.0, -1.0, 0.5];
    /// let result = data.acos(); // [0.0, π/2, π, π/3]
    /// ```
    fn acos(&self) -> Self::Output;

    /// Computes the arcsine (inverse sine) of each element.
    ///
    /// Returns the angle in radians whose sine is the input value.
    /// Valid input domain: [-1, 1], output range: [-π/2, π/2].
    ///
    /// # Domain
    ///
    /// - Input: [-1, 1]
    /// - Output: [-π/2, π/2] radians
    /// - Invalid inputs (|x| > 1) return NaN
    fn asin(&self) -> Self::Output;

    /// Computes the arctangent (inverse tangent) of each element.
    ///
    /// Returns the angle in radians whose tangent is the input value.
    /// Valid for all finite inputs, output range: (-π/2, π/2).
    ///
    /// # Domain
    ///
    /// - Input: All real numbers
    /// - Output: (-π/2, π/2) radians
    fn atan(&self) -> Self::Output;

    /// Computes the two-argument arctangent of each element pair.
    ///
    /// Computes atan2(y, x) where `self` represents the y-coordinates
    /// and `other` represents the x-coordinates. Returns the angle in radians
    /// between the positive x-axis and the point (x, y).
    /// Output range: [-π, π].
    ///
    /// # Parameters
    ///
    /// - `self` (y): Y-coordinates
    /// - `other` (x): X-coordinates
    ///
    /// # Domain
    ///
    /// - Input: All real number pairs (y, x)
    /// - Output: [-π, π] radians
    /// - Handles signs correctly for all quadrants
    fn atan2(&self, other: Self) -> Self::Output;

    /// Computes the cube root of each element.
    ///
    /// Returns the value that, when cubed, gives the input value.
    /// Unlike square root, cube root is defined for negative numbers.
    ///
    /// # Domain
    ///
    /// - Input: All real numbers
    /// - Output: All real numbers
    /// - Preserves sign: cbrt(-x) = -cbrt(x)
    fn cbrt(&self) -> Self::Output;

    /// Computes the floor (round down) of each element.
    ///
    /// Returns the largest integer less than or equal to the input value.
    ///
    /// # Examples
    ///
    /// - floor(2.7) = 2.0
    /// - floor(-1.3) = -2.0
    /// - floor(3.0) = 3.0
    fn floor(&self) -> Self::Output;

    /// Computes the natural exponential (e^x) of each element.
    ///
    /// Returns e raised to the power of each input element.
    /// Result grows very rapidly for positive inputs.
    ///
    /// # Domain
    ///
    /// - Input: All real numbers
    /// - Output: (0, ∞) for finite inputs
    /// - exp(±∞) = ∞/0, exp(NaN) = NaN
    fn exp(&self) -> Self::Output;

    /// Computes the natural logarithm of each element.
    ///
    /// Returns the logarithm base e of each input element.
    /// Only defined for positive inputs.
    ///
    /// # Domain
    ///
    /// - Input: (0, ∞)
    /// - Output: (-∞, ∞)
    /// - ln(x) = NaN for x ≤ 0
    fn ln(&self) -> Self::Output;

    /// Computes the sine of each element.
    ///
    /// Returns the sine of each input element (in radians). Uses optimized range reduction
    /// for large inputs to maintain accuracy across the full input domain.
    ///
    /// # Domain & Range
    /// - **Input**: All real numbers (radians)
    /// - **Output**: [-1, 1]
    /// - **Accuracy**: Typically within 1-2 ULP of standard library
    ///
    /// # Performance
    /// - **Implementation**: Range reduction + polynomial approximation
    /// - **Typical speedup**: 4-8x over scalar (depending on input range)
    /// - **Best performance**: For inputs in [-π, π] range
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    /// use std::f32::consts::PI;
    ///
    /// let angles = vec![0.0, PI/2.0, PI, 3.0*PI/2.0];
    /// let result = angles.sin(); // [0.0, 1.0, 0.0, -1.0]
    /// ```
    fn sin(&self) -> Self::Output;

    /// Computes the cosine of each element.
    ///
    /// Returns the cosine of each input element (in radians). Uses optimized range reduction
    /// for large inputs to maintain accuracy across the full input domain.
    ///
    /// # Domain & Range
    /// - **Input**: All real numbers (radians)
    /// - **Output**: [-1, 1]
    /// - **Accuracy**: Typically within 1-2 ULP of standard library
    ///
    /// # Performance
    /// - **Implementation**: Range reduction + polynomial approximation
    /// - **Typical speedup**: 4-13x over scalar (function showing best SIMD gains)
    /// - **Best performance**: For inputs in [-π, π] range
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    /// use std::f32::consts::PI;
    ///
    /// let angles = vec![0.0, PI/2.0, PI, 3.0*PI/2.0];
    /// let result = angles.cos(); // [1.0, 0.0, -1.0, 0.0]
    /// ```
    fn cos(&self) -> Self::Output;

    /// Computes the tangent of each element.
    ///
    /// Returns the tangent of each input element (in radians).
    /// Becomes infinite at odd multiples of π/2.
    ///
    /// # Domain
    ///
    /// - Input: All real numbers except odd multiples of π/2
    /// - Output: All real numbers
    fn tan(&self) -> Self::Output;

    /// Computes the square root of each element.
    ///
    /// Returns the positive square root of each input element using hardware-accelerated
    /// SIMD instructions where available.
    ///
    /// # Domain & Range
    /// - **Input**: [0, ∞) (negative values return NaN)
    /// - **Output**: [0, ∞)
    /// - **Accuracy**: Typically within 0.5 ULP (hardware precision)
    ///
    /// # Performance
    /// - **Implementation**: Native SIMD square root instructions
    /// - **Typical speedup**: 4-8x over scalar
    /// - **Hardware support**: Uses `_mm256_sqrt_ps` (AVX2) or `vsqrtq_f32` (NEON)
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    ///
    /// let data = vec![4.0, 9.0, 16.0, 25.0];
    /// let result = data.sqrt(); // [2.0, 3.0, 4.0, 5.0]
    /// ```
    fn sqrt(&self) -> Self::Output;

    /// Computes the ceiling (round up) of each element.
    ///
    /// Returns the smallest integer greater than or equal to the input value.
    ///
    /// # Examples
    ///
    /// - ceil(2.3) = 3.0
    /// - ceil(-1.7) = -1.0
    /// - ceil(4.0) = 4.0
    fn ceil(&self) -> Self::Output;

    /// Computes x raised to the power y for each element pair.
    ///
    /// Returns x^y for corresponding elements.
    /// Handles special cases according to IEEE 754 standards.
    ///
    /// # Domain
    ///
    /// Complex rules apply for negative bases and fractional exponents.
    /// Consult IEEE 754 for complete specification.
    fn pow(&self, other: Self) -> Self::Output;

    /// Computes fused multiply-add (FMA) operation: a * b + c.
    ///
    /// Performs `self * multiplier + addend` for corresponding elements.
    /// FMA is computed as a single operation with higher precision than
    /// separate multiply and add operations.
    ///
    /// # Parameters
    ///
    /// - `self`: Addend (value to add to the product)
    /// - `multiplier`: First multiplicand
    /// - `multiplicand`: Second multiplicand
    ///
    /// # Performance
    ///
    /// - **Implementation**: Native FMA instruction when available (AVX2/FMA3, NEON)
    /// - **Precision**: Higher precision than separate multiply-add
    /// - **Typical speedup**: 1.5-2x over separate multiply-add operations
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    ///
    /// let multiplier = vec![2.0, 3.0, 4.0, 5.0];     
    /// let multiplicand = vec![1.5, 2.5, 3.5, 4.5];       
    /// let addend = vec![1.0, 1.0, 1.0, 1.0];         // self
    /// let result = addend.fma(multiplier, multiplicand); // [4.0, 8.5, 15.0, 23.5]
    /// ```
    fn fma(&self, multiplier: Self, multiplicand: Self) -> Self::Output;

    /// Computes the Euclidean distance (2D hypotenuse) for element pairs.
    ///
    /// Returns sqrt(x² + y²) where `self` represents the x-coordinates
    /// and `other` represents the y-coordinates. Computed in a numerically
    /// stable way that avoids overflow for large values and underflow for small values.
    ///
    /// # Parameters
    ///
    /// - `self` (x): X-coordinates
    /// - `other` (y): Y-coordinates
    ///
    /// # Algorithm
    /// Uses scaling and normalization to prevent intermediate overflow/underflow,
    /// maintaining accuracy across the full range of f32 values.
    ///
    /// # Performance
    /// - **Implementation**: Optimized scaling + SIMD square root
    /// - **Typical speedup**: 2-4x over scalar
    /// - **Stability**: Handles extreme values (1e-20 to 1e20) safely
    ///
    /// # Examples
    /// ```rust
    /// use simdly::simd::SimdMath;
    ///
    /// let x = vec![3.0, 5.0, 1e20];
    /// let y = vec![4.0, 12.0, 1e20];
    /// let distances = x.hypot(y); // [5.0, 13.0, ~1.41e20]
    /// ```
    fn hypot(&self, other: Self) -> Self::Output;

    /// Computes the 3D Euclidean distance for element triplets.
    ///
    /// Returns sqrt(x² + y² + z²) where `self` is x, `other1` is y, and `other2` is z.
    /// Computed with care to avoid intermediate overflow/underflow.
    ///
    /// # Parameters
    ///
    /// - `self` (x): X-coordinates
    /// - `other1` (y): Y-coordinates  
    /// - `other2` (z): Z-coordinates
    fn hypot3(&self, other1: Self, other2: Self) -> Self::Output;

    /// Computes the 4D Euclidean distance for element quadruplets.
    ///
    /// Returns sqrt(x² + y² + z² + w²) where `self` is x, `other1` is y,
    /// `other2` is z, and `other3` is w. Computed with care to avoid
    /// intermediate overflow/underflow.
    ///
    /// # Parameters
    ///
    /// - `self` (x): X-coordinates
    /// - `other1` (y): Y-coordinates
    /// - `other2` (z): Z-coordinates
    /// - `other3` (w): W-coordinates
    fn hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output;

    // ================================================================================================
    // PARALLEL SIMD METHODS
    // ================================================================================================

    /// Computes the absolute value of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing
    /// for maximum performance on multi-core systems.
    ///
    /// # Platform Support
    ///
    /// - **x86_64**: Uses AVX2 instructions with 8-element vectors (256-bit)
    /// - **aarch64**: Uses NEON instructions with 4-element vectors (128-bit)
    /// - **Other platforms**: Falls back to scalar operations
    fn par_abs(&self) -> Self::Output;

    /// Computes the arccosine of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_acos(&self) -> Self::Output;

    /// Computes the arcsine of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_asin(&self) -> Self::Output;

    /// Computes the arctangent of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_atan(&self) -> Self::Output;

    /// Computes the two-argument arctangent using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_atan2(&self, other: Self) -> Self::Output;

    /// Computes the cube root of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_cbrt(&self) -> Self::Output;

    /// Computes the ceiling of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_ceil(&self) -> Self::Output;

    /// Computes the cosine of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing
    /// for maximum performance on multi-core systems.
    ///
    /// # Platform Support & Performance
    ///
    /// ## AVX2 (x86_64)
    /// - **Vector width**: 8 elements (256-bit)
    /// - **Instructions**: Uses optimized AVX2 intrinsics
    /// - **Performance**: Up to 13.3x faster than scalar for large arrays
    /// - **Parallel chunks**: 8192 elements per thread
    ///
    /// ## NEON (aarch64)  
    /// - **Vector width**: 4 elements (128-bit)
    /// - **Instructions**: Uses optimized NEON intrinsics
    /// - **Performance**: Significant speedup on ARM processors
    /// - **Parallel chunks**: 4096 elements per thread
    ///
    /// ## Fallback
    /// - **Other platforms**: Scalar implementation with compiler auto-vectorization
    ///
    /// # Threshold Selection
    /// - Arrays ≤ `PARALLEL_SIMD_THRESHOLD`: Uses single-threaded SIMD
    /// - Arrays > `PARALLEL_SIMD_THRESHOLD`: Uses multi-threaded parallel SIMD
    fn par_cos(&self) -> Self::Output;

    /// Computes the exponential function of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_exp(&self) -> Self::Output;

    /// Computes the floor of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_floor(&self) -> Self::Output;

    /// Computes the natural logarithm of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_ln(&self) -> Self::Output;

    /// Computes the sine of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing
    /// for maximum performance on multi-core systems.
    fn par_sin(&self) -> Self::Output;

    /// Computes the square root of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_sqrt(&self) -> Self::Output;

    /// Computes the tangent of each element using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_tan(&self) -> Self::Output;

    /// Computes the Euclidean distance (2D hypotenuse) using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_hypot(&self, other: Self) -> Self::Output;

    /// Computes the 3D Euclidean distance using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_hypot3(&self, other1: Self, other2: Self) -> Self::Output;

    /// Computes the 4D Euclidean distance using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output;

    /// Computes x raised to the power y using parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_pow(&self, other: Self) -> Self::Output;

    /// Computes fused multiply-add (FMA) operation using parallel SIMD.
    ///
    /// Performs `multiplier * multiplicand + self` using size-adaptive parallel SIMD.
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    /// For arrays larger than PARALLEL_SIMD_THRESHOLD, uses multi-threaded processing.
    fn par_fma(&self, multiplier: Self, multiplicand: Self) -> Self::Output;
}

/// Trait for SIMD data shuffling and permutation operations.
///
/// This trait provides efficient methods for rearranging elements within SIMD vectors,
/// including broadcasting individual elements and swapping between vector halves.
/// These operations are fundamental building blocks for more complex SIMD algorithms
/// like matrix operations, convolutions, and data transformations.
///
/// # Platform Support
///
/// - **x86/x86_64**: Uses AVX2 permute instructions (`_mm256_permute_ps`, `_mm256_permute2f128_ps`)
/// - **ARM/AArch64**: Uses NEON shuffle and permute instructions
/// - **Fallback**: Scalar implementations with manual element copying
///
/// # Performance Characteristics
///
/// - **Latency**: Typically 1-3 cycles for most shuffle operations
/// - **Throughput**: Can execute multiple shuffle operations per cycle on modern CPUs
/// - **Zero data movement**: Operations work entirely within vector registers
/// - **Compile-time masks**: Template parameters enable optimal instruction selection
///
/// # Usage Examples
///
/// ## Element Broadcasting
/// ```rust
/// use simdly::simd::avx2::f32x8::F32x8;
/// use simdly::simd::SimdShuffle;
///
/// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
///
/// // Broadcast first element to all positions: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
/// let broadcast = vec.permute::<0x00>();
///
/// // Broadcast second element: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  
/// let broadcast2 = vec.permute::<0x55>();
/// ```
///
/// ## Lane Swapping
/// ```rust
/// use simdly::simd::avx2::f32x8::F32x8;
/// use simdly::simd::SimdShuffle;
///
/// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
///
/// // Duplicate lower 4 elements: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
/// let low_dup = vec.permute2f128::<0x00>();
///
/// // Duplicate upper 4 elements: [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
/// let high_dup = vec.permute2f128::<0x11>();
/// ```
///
/// # Implementation Requirements
///
/// Implementations should:
/// - Use compile-time constant masks for optimal performance
/// - Leverage the fastest available shuffle instructions for the target platform  
/// - Maintain element precision without introducing rounding errors
/// - Handle all bit patterns correctly including special values (NaN, infinity)
/// - Provide consistent behavior across different architectures
pub trait SimdShuffle {
    /// The output type returned by shuffle operations
    type Output;

    /// Permutes elements within each 128-bit lane using a compile-time mask.
    ///
    /// This operation rearranges elements within the lower and upper 128-bit halves
    /// of the vector independently. Each 4-bit group in the mask selects which
    /// source element to copy to the corresponding output position.
    ///
    /// # Mask Encoding
    ///
    /// The 8-bit mask parameter encodes 4 selections, each 2 bits wide:
    /// - Bits \[1:0\] → Output element 0 source
    /// - Bits \[3:2\] → Output element 1 source
    /// - Bits \[5:4\] → Output element 2 source  
    /// - Bits \[7:6\] → Output element 3 source
    ///
    /// Each 2-bit value selects from positions 0, 1, 2, or 3 within the same 128-bit lane.
    ///
    /// # Common Mask Values
    ///
    /// - `0x00` (00_00_00_00): Broadcast element 0 → [a, a, a, a, e, e, e, e]
    /// - `0x55` (01_01_01_01): Broadcast element 1 → [b, b, b, b, f, f, f, f]  
    /// - `0xAA` (10_10_10_10): Broadcast element 2 → [c, c, c, c, g, g, g, g]
    /// - `0xFF` (11_11_11_11): Broadcast element 3 → [d, d, d, d, h, h, h, h]
    /// - `0x1B` (00_01_10_11): Reverse 4 elements → [d, c, b, a, h, g, f, e]
    ///
    /// # Arguments
    ///
    /// * `MASK` - Compile-time 8-bit mask specifying the permutation pattern
    ///
    /// # Returns
    ///
    /// A new vector with elements permuted according to the mask within each 128-bit lane.
    ///
    /// # Performance
    ///
    /// - **Latency**: 1-3 cycles depending on CPU microarchitecture
    /// - **Throughput**: 1-2 operations per cycle on modern processors
    /// - **Ports**: Typically executes on shuffle/permute execution ports
    ///
    /// # Platform Implementation
    ///
    /// - **AVX2**: Uses `_mm256_permute_ps` intrinsic
    /// - **NEON**: Uses combination of `vrev`, `vdup`, and `vtbl` instructions
    /// - **Scalar**: Manual element copying with compiler optimization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdly::simd::SimdShuffle;
    /// # use simdly::simd::avx2::f32x8::F32x8;
    ///
    /// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
    ///
    /// // Broadcast first element in each lane
    /// let broadcast = vec.permute::<0x00>();  // [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
    ///
    /// // Reverse elements within each lane  
    /// let reversed = vec.permute::<0x1B>();   // [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]
    /// ```
    fn permute<const MASK: i32>(&self) -> Self::Output;

    /// Permutes 128-bit lanes within the 256-bit vector using a compile-time mask.
    ///
    /// This operation swaps or duplicates the upper and lower 128-bit halves of
    /// the vector. It's essential for operations that need to move data between
    /// the two halves of AVX2 vectors.
    ///
    /// # Mask Encoding
    ///
    /// The 8-bit mask parameter controls source selection for each output lane:
    /// - Bits \[3:0\] → Lower 128-bit lane source (0 = input\[127:0\], 1 = input\[255:128\])  
    /// - Bits \[7:4\] → Upper 128-bit lane source (0 = input\[127:0\], 1 = input\[255:128\])
    ///
    /// # Common Mask Values
    ///
    /// - `0x00`: Duplicate lower lane → [a, b, c, d, a, b, c, d]
    /// - `0x11`: Duplicate upper lane → [e, f, g, h, e, f, g, h]  
    /// - `0x01`: Swap lanes → [e, f, g, h, a, b, c, d]
    /// - `0x10`: Identity (no change) → [a, b, c, d, e, f, g, h]
    ///
    /// # Arguments
    ///
    /// * `MASK` - Compile-time 8-bit mask specifying which lanes to select
    ///
    /// # Returns
    ///
    /// A new vector with 128-bit lanes rearranged according to the mask.
    ///
    /// # Performance
    ///
    /// - **Latency**: 1-3 cycles (may be higher for cross-lane operations)
    /// - **Throughput**: 0.5-1 operations per cycle depending on mask complexity
    /// - **Domain crossing**: Some masks may cause execution domain penalties
    ///
    /// # Platform Implementation
    ///
    /// - **AVX2**: Uses `_mm256_permute2f128_ps` intrinsic  
    /// - **NEON**: Emulated using combinations of `vget_low`/`vget_high` and `vcombine`
    /// - **Scalar**: Manual 4-element copying with compiler optimization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdly::simd::SimdShuffle;
    /// # use simdly::simd::avx2::f32x8::F32x8;
    ///
    /// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
    ///
    /// // Duplicate lower 4 elements
    /// let low_dup = vec.permute2f128::<0x00>();   // [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    ///
    /// // Duplicate upper 4 elements
    /// let high_dup = vec.permute2f128::<0x11>();  // [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
    ///
    /// // Swap the two halves
    /// let swapped = vec.permute2f128::<0x01>();   // [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    /// ```
    fn permute2f128<const MASK: i32>(&self) -> Self::Output;
}
