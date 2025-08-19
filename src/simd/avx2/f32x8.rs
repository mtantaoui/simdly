//! AVX2 8-lane f32 SIMD vector implementation.
//!
//! This module provides `F32x8`, a SIMD vector type that wraps Intel's AVX2 `__m256`
//! intrinsic to perform vectorized operations on 8 single-precision floating-point
//! values simultaneously using 256-bit AVX2 instructions.
//!
//! # Architecture Requirements
//!
//! - **CPU Support**: Intel processors with AVX2 support (Haswell and later)
//! - **Target Architecture**: x86_64 (and x86 with SSE compatibility)
//! - **Compilation**: Must be compiled with AVX2 enabled (`-C target-feature=+avx2`)
//!
//! # Performance Characteristics
//!
//! - **Vector Width**: 256 bits (8 × f32)
//! - **Memory Alignment**: Optimal performance with 32-byte aligned data
//! - **Throughput**: Up to 8× speedup for vectorizable operations compared to scalar code
//! - **Power Efficiency**: High performance for server and desktop processors
//!
//! # Supported Operations
//!
//! ## Loading and Storing
//! - `From<&[f32]>` trait - High-level loading with automatic partial handling
//! - `load_aligned()`, `load_unaligned()` - Direct memory loading
//! - `load_partial()` - Safe partial loading for sizes < 8
//! - `store_at()` - Automatic store with size detection
//! - `store_aligned_at()`, `store_unaligned_at()` - Direct memory storing
//!
//! ## Mathematical Functions
//! - **Basic**: `abs()`, `sqrt()`, `floor()`, `ceil()`
//! - **Trigonometric**: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`
//! - **Exponential**: `exp()`, `ln()`, `cbrt()`
//! - **Distance**: `hypot()`, `hypot3()`, `hypot4()`
//!
//! ## Arithmetic Operators
//! - Element-wise addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`)
//! - All operators support operation chaining for complex expressions

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::ops::{Add, Div, Mul, Sub};

use crate::simd::{avx2::math::*, Alignment, SimdLoad, SimdMath, SimdShuffle, SimdStore};

/// AVX2 memory alignment requirement in bytes.
///
/// AVX2 operations perform optimally when data is aligned to 32-byte boundaries.
/// This constant defines the alignment requirement for F32x8 vectors to achieve
/// maximum performance with `_mm256_load_ps` and `_mm256_store_ps` instructions.
pub(crate) const AVX_ALIGNMENT: usize = 32;

/// Number of f32 elements that fit in an AVX2 256-bit vector.
///
/// AVX2 vectors can contain 8 single-precision floating-point values
/// (8 × 32 bits = 256 bits). This constant defines the vector capacity
/// and is used for bounds checking and loop unrolling optimizations.
pub(crate) const LANE_COUNT: usize = 8;

/// AVX2 SIMD vector containing 8 packed f32 values.
///
/// This structure provides efficient vectorized operations on 8 single-precision
/// floating-point numbers using AVX2 instructions. It maintains both the underlying
/// AVX2 register and the count of valid elements for partial operations.
///
/// # Memory Alignment
///
/// For optimal performance, data should be aligned to 32-byte boundaries when possible.
/// AVX2 instructions can handle unaligned data but aligned access is faster.
///
/// # Usage
///
/// ## Basic Loading and Storing
/// ```rust
/// # #[cfg(target_feature = "avx2")]
/// # {
/// use simdly::simd::avx2::f32x8::F32x8;
/// use simdly::simd::SimdLoad;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let vector = F32x8::from(data.as_slice());
/// # }
/// ```
///
/// ## Mathematical Operations
/// ```rust
/// # use simdly::simd::avx2::f32x8::{self, F32x8};
/// # use simdly::simd::{SimdLoad, SimdMath};
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let vec = F32x8::from(data.as_slice());
/// // Compute square root of all elements simultaneously
/// let sqrt_vec = vec.sqrt();
///
/// // Compute sine of all elements
/// let sin_vec = vec.sin();
///
/// // Chain operations
/// let result = vec.abs().sqrt().sin();
/// ```
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    /// Number of valid elements in the vector (1-8)
    pub size: usize,
    /// AVX2 256-bit vector register containing 8 packed f32 values
    pub elements: __m256,
}

impl Alignment<f32> for F32x8 {
    /// Checks if a pointer is properly aligned for AVX2 operations.
    ///
    /// AVX2 operations perform optimally when data is aligned to 32-byte boundaries.
    /// This function checks if the given pointer meets this alignment requirement.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to check for alignment
    ///
    /// # Returns
    ///
    /// `true` if the pointer is 32-byte aligned, `false` otherwise
    #[inline(always)]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;

        ptr % core::mem::align_of::<__m256>() == 0
    }
}

impl From<&[f32]> for F32x8 {
    /// Creates an F32x8 vector from a slice of f32 values.
    ///
    /// Automatically selects the appropriate loading method based on slice length:
    /// - For slices with exactly 8 elements: Uses full SIMD load
    /// - For slices with fewer than 8 elements: Uses partial load with zero-padding
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdly::simd::avx2::f32x8::F32x8;
    /// 
    /// let full_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let vec = F32x8::from(full_data.as_slice());
    /// 
    /// let partial_data = [1.0, 2.0, 3.0];
    /// let partial_vec = F32x8::from(partial_data.as_slice());
    /// ```
    fn from(slice: &[f32]) -> Self {
        debug_assert!(!slice.is_empty(), "data pointer can't be NULL");

        let size = slice.len();

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), size) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }
}

impl SimdLoad<f32> for F32x8 {
    type Output = Self;

    // /// Loads data from a slice into the SIMD vector.
    // ///
    // /// This is the high-level interface for loading f32 data. It automatically
    // /// handles partial loads for slices smaller than 8 elements and uses the
    // /// most appropriate loading strategy based on slice size.
    // ///
    // /// # Arguments
    // ///
    // /// * `slice` - Input slice of f32 values
    // ///
    // /// # Returns
    // ///
    // /// F32x8 instance with loaded data
    // ///
    // /// # Behavior
    // ///
    // /// - For slices < 8 elements: Uses masked partial loading
    // /// - For slices >= 8 elements: Loads the first 8 elements using full loading
    // ///
    // /// # Panics
    // ///
    // /// Panics in debug builds if the slice is empty.
    // #[inline(always)]
    // fn from_slice(slice: &[f32]) -> Self::Output {
    //     debug_assert!(!slice.is_empty(), "data pointer can't be NULL");

    //     let size = slice.len();

    //     match slice.len().cmp(&LANE_COUNT) {
    //         std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), size) },
    //         std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
    //             Self::load(slice.as_ptr(), LANE_COUNT)
    //         },
    //     }
    // }

    /// Loads exactly 8 elements from memory.
    ///
    /// Automatically chooses between aligned and unaligned load based on pointer alignment.
    /// This function is optimized for loading complete vectors.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Must be exactly 8
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least 8 valid f32 values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size != 8 or if pointer is null.
    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F32x8::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads 8 elements from 32-byte aligned memory.
    ///
    /// This is the fastest loading method when alignment is guaranteed.
    /// Uses the optimized `_mm256_load_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - 32-byte aligned pointer to f32 data
    ///
    /// # Safety
    ///
    /// Pointer must be 32-byte aligned and point to at least 8 valid f32 values.
    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: _mm256_load_ps(ptr),
            size: LANE_COUNT,
        }
    }

    /// Loads 8 elements from unaligned memory.
    ///
    /// Slightly slower than aligned load but works with any memory alignment.
    /// Uses the `_mm256_loadu_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 8 valid f32 values.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: _mm256_loadu_ps(ptr),
            size: LANE_COUNT,
        }
    }

    /// Loads fewer than 8 elements using masked loading operations.
    ///
    /// This function safely loads partial data when the source contains fewer than
    /// 8 elements. It uses AVX2 masked load operations to prevent reading beyond
    /// the valid data range.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Number of elements to load (1-7)
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least `size` valid f32 values.
    ///
    /// # Implementation
    ///
    /// Uses `_mm256_maskload_ps` with dynamically generated masks based on the
    /// size parameter. Unloaded lanes contain undefined values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size >= 8 or if pointer is null.
    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );

        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask = match size {
            1 => _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0),
            2 => _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
            4 => _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            5 => _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0),
            6 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
            7 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0),
            _ => unreachable!(),
        };

        Self {
            elements: _mm256_maskload_ps(ptr, mask),
            size,
        }
    }
}

impl SimdMath for F32x8 {
    type Output = Self;

    /// Computes the absolute value of each element using AVX2 intrinsics.
    ///
    /// Uses `_mm256_abs_ps` to efficiently clear the sign bit of all elements.
    #[inline(always)]
    fn abs(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_abs_ps(self.elements) },
        }
    }

    /// Computes the arccosine using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_acos_ps` function from the math module for vectorized
    /// arccosine computation with polynomial approximation.
    #[inline(always)]
    fn acos(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_acos_ps(self.elements) },
        }
    }

    /// Computes the arcsine using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_asin_ps` function from the math module for vectorized
    /// arcsine computation with polynomial approximation.
    #[inline(always)]
    fn asin(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_asin_ps(self.elements) },
        }
    }

    /// Computes the arctangent using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_atan_ps` function from the math module for vectorized
    /// arctangent computation with polynomial approximation.
    #[inline(always)]
    fn atan(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_atan_ps(self.elements) },
        }
    }

    /// Computes two-argument arctangent using optimized AVX2 math functions.
    ///
    /// This implementation treats the vector as pairs of (y, x) coordinates.
    /// Uses the `_mm256_atan2_ps` function for proper quadrant handling.
    #[inline(always)]
    fn atan2(&self, other: Self) -> Self::Output {
        unsafe {
            // self is y, other is x (following atan2(y, x) convention)
            let y_vec = self.elements;
            let x_vec = other.elements;
            Self {
                elements: _mm256_atan2_ps(y_vec, x_vec),
                size: self.size,
            }
        }
    }

    /// Computes the cube root using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_cbrt_ps` function from the math module for vectorized
    /// cube root computation with Newton-Raphson iteration.
    #[inline(always)]
    fn cbrt(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_cbrt_ps(self.elements) },
        }
    }

    /// Computes the floor using AVX2's native rounding intrinsic.
    ///
    /// Uses `_mm256_floor_ps` for efficient vectorized floor operation.
    #[inline(always)]
    fn floor(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_floor_ps(self.elements) },
        }
    }

    /// Computes the natural exponential using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_exp_ps` function from the math module for vectorized
    /// exponential computation with range reduction and polynomial approximation.
    #[inline(always)]
    fn exp(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_exp_ps(self.elements) },
        }
    }

    /// Computes the natural logarithm using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_ln_ps` function from the math module for vectorized
    /// logarithm computation with range reduction and polynomial approximation.
    #[inline(always)]
    fn ln(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_ln_ps(self.elements) },
        }
    }

    /// Computes 2D Euclidean distance using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_hypot_ps` function for numerically stable computation.
    #[inline(always)]
    fn hypot(&self, other: Self) -> Self::Output {
        unsafe {
            Self {
                elements: _mm256_hypot_ps(self.elements, other.elements),
                size: self.size,
            }
        }
    }

    /// Computes x raised to the power y using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_pow_ps` function for vectorized power computation.
    #[inline(always)]
    fn pow(&self, other: Self) -> Self::Output {
        unsafe {
            let x_vec = self.elements;
            let y_vec = other.elements;
            Self {
                elements: _mm256_pow_ps(x_vec, y_vec),
                size: self.size,
            }
        }
    }

    /// Computes sine using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_sin_ps` function from the math module for vectorized
    /// sine computation with range reduction and polynomial approximation.
    #[inline(always)]
    fn sin(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_sin_ps(self.elements) },
        }
    }

    /// Computes cosine using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_cos_ps` function from the math module for vectorized
    /// cosine computation with range reduction and polynomial approximation.
    #[inline(always)]
    fn cos(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_cos_ps(self.elements) },
        }
    }

    /// Computes tangent using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_tan_ps` function from the math module for vectorized
    /// tangent computation with range reduction and polynomial approximation.
    #[inline(always)]
    fn tan(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_tan_ps(self.elements) },
        }
    }

    /// Computes square root using AVX2's native square root intrinsic.
    ///
    /// Uses `_mm256_sqrt_ps` for efficient vectorized square root computation.
    #[inline(always)]
    fn sqrt(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_sqrt_ps(self.elements) },
        }
    }

    /// Computes the ceiling using AVX2's native rounding intrinsic.
    ///
    /// Uses `_mm256_ceil_ps` for efficient vectorized ceiling operation.
    #[inline(always)]
    fn ceil(&self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_ceil_ps(self.elements) },
        }
    }

    /// Computes 3D Euclidean distance using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_hypot3_ps` function for numerically stable computation.
    #[inline(always)]
    fn hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        unsafe {
            Self {
                elements: _mm256_hypot3_ps(self.elements, other1.elements, other2.elements),
                size: self.size,
            }
        }
    }

    /// Computes 4D Euclidean distance using optimized AVX2 math functions.
    ///
    /// Uses the `_mm256_hypot4_ps` function for numerically stable computation.
    #[inline(always)]
    fn hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        unsafe {
            Self {
                elements: _mm256_hypot4_ps(
                    self.elements,
                    other1.elements,
                    other2.elements,
                    other3.elements,
                ),
                size: self.size,
            }
        }
    }

    // ================================================================================================
    // PARALLEL SIMD METHODS
    // ================================================================================================
    // For individual vectors (8 elements), parallel methods delegate to regular methods
    // since the data size is too small to benefit from multi-threading overhead.

    /// Computes absolute value. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_abs(&self) -> Self::Output {
        self.abs()
    }

    /// Computes arccosine. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_acos(&self) -> Self::Output {
        self.acos()
    }

    /// Computes arcsine. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_asin(&self) -> Self::Output {
        self.asin()
    }

    /// Computes arctangent. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_atan(&self) -> Self::Output {
        self.atan()
    }

    /// Computes two-argument arctangent. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_atan2(&self, other: Self) -> Self::Output {
        self.atan2(other)
    }

    /// Computes cube root. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_cbrt(&self) -> Self::Output {
        self.cbrt()
    }

    /// Computes ceiling. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_ceil(&self) -> Self::Output {
        self.ceil()
    }

    /// Computes cosine. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_cos(&self) -> Self::Output {
        self.cos()
    }

    /// Computes exponential. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_exp(&self) -> Self::Output {
        self.exp()
    }

    /// Computes floor. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_floor(&self) -> Self::Output {
        self.floor()
    }

    /// Computes natural logarithm. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_ln(&self) -> Self::Output {
        self.ln()
    }

    /// Computes sine. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_sin(&self) -> Self::Output {
        self.sin()
    }

    /// Computes square root. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_sqrt(&self) -> Self::Output {
        self.sqrt()
    }

    /// Computes tangent. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_tan(&self) -> Self::Output {
        self.tan()
    }

    /// Computes 2D Euclidean distance. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_hypot(&self, other: Self) -> Self::Output {
        self.hypot(other)
    }

    /// Computes 3D Euclidean distance. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        self.hypot3(other1, other2)
    }

    /// Computes 4D Euclidean distance. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        self.hypot4(other1, other2, other3)
    }

    /// Computes power function. Delegates to regular method for individual vectors.
    #[inline(always)]
    fn par_pow(&self, other: Self) -> Self::Output {
        self.pow(other)
    }
}

impl SimdStore<f32> for F32x8 {
    type Output = Self;

    /// Stores vector data at the given pointer location.
    ///
    /// Automatically chooses the most appropriate store method based on:
    /// - Vector size (partial vs. full store)
    /// - Pointer alignment (aligned vs. unaligned store)
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to sufficient writable memory
    /// for `self.size` elements.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size > 8 or if pointer is null.
    #[inline(always)]
    fn store_at(&self, ptr: *const f32) {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Cast to mutable pointer for store operations
        let mut_ptr = ptr as *mut f32;

        match self.size.cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { self.store_at_partial(mut_ptr) },
            std::cmp::Ordering::Equal => match F32x8::is_aligned(ptr) {
                true => unsafe { self.store_aligned_at(mut_ptr) },
                false => unsafe { self.store_unaligned_at(mut_ptr) },
            },
            std::cmp::Ordering::Greater => unreachable!("Size cannot exceed LANE_COUNT"),
        }
    }

    /// Non-temporal store that bypasses cache.
    ///
    /// Stores all 8 vector elements to memory without polluting the cache.
    /// This is optimal for large datasets where data won't be reused soon.
    ///
    /// # Arguments
    ///
    /// * `ptr` - **32-byte aligned** mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// - Pointer must be 32-byte aligned (required for correctness, not just performance)
    /// - Pointer must point to at least 8 valid f32 memory locations
    /// - Use `Alignment::is_aligned()` to verify alignment before calling
    ///
    /// # Performance Notes
    ///
    /// - **Alignment Required**: The underlying `_mm256_stream_ps` intrinsic requires 32-byte alignment
    /// - **Cache Bypass**: Data is written directly to memory, bypassing CPU cache
    /// - **Memory Ordering**: May have weaker memory ordering guarantees than regular stores
    ///
    /// # Use Case
    ///
    /// Best for write-once scenarios with large datasets that won't be
    /// accessed again soon, such as:
    /// - Large array initialization
    /// - Data export operations  
    /// - Streaming computations with sequential access patterns
    #[inline(always)]
    unsafe fn stream_at(&self, ptr: *mut f32) {
        _mm256_stream_ps(ptr, self.elements)
    }

    /// Stores 8 elements to 32-byte aligned memory.
    ///
    /// This is the fastest store method when alignment is guaranteed.
    /// Uses the optimized `_mm256_store_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - 32-byte aligned mutable pointer to destination
    ///
    /// # Safety
    ///
    /// Pointer must be 32-byte aligned and point to at least 8 valid f32
    /// memory locations.
    #[inline(always)]
    unsafe fn store_aligned_at(&self, ptr: *mut f32) {
        _mm256_store_ps(ptr, self.elements)
    }

    /// Stores 8 elements to unaligned memory.
    ///
    /// Works with any memory alignment but slightly slower than aligned store.
    /// Uses the `_mm256_storeu_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 8 valid f32 memory locations.
    #[inline(always)]
    unsafe fn store_unaligned_at(&self, ptr: *mut f32) {
        _mm256_storeu_ps(ptr, self.elements)
    }

    /// Stores only the valid elements using masked store operations.
    ///
    /// This function safely stores partial data when the vector contains fewer
    /// than 8 valid elements. It uses AVX2 masked store operations to prevent
    /// writing beyond the intended memory range.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least `self.size` valid
    /// f32 memory locations.
    ///
    /// # Implementation
    ///
    /// Uses `_mm256_maskstore_ps` with masks corresponding to `self.size`.
    /// Only the first `self.size` elements are written to memory.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size >= 8 or if pointer is null.
    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(
            self.size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __m256i = match self.size {
            1 => _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0),
            2 => _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
            4 => _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            5 => _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0),
            6 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
            7 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0),
            _ => unreachable!("Size must be < LANE_COUNT"),
        };

        _mm256_maskstore_ps(ptr, mask, self.elements);
    }
}

/// Implementation of SIMD shuffle and permutation operations for F32x8 vectors.
///
/// This implementation provides efficient element rearrangement operations using
/// AVX2 permute instructions, enabling high-performance data shuffling within
/// 256-bit vectors without moving data outside of vector registers.
///
/// # Performance Characteristics
///
/// - **Latency**: 1-3 cycles for most permutation patterns
/// - **Throughput**: 1-2 operations per cycle on modern Intel/AMD processors  
/// - **Execution**: Runs on dedicated shuffle/permute execution ports
/// - **Zero Memory**: Operations work entirely within vector registers
///
/// # Implementation Details
///
/// Uses AVX2 intrinsics:
/// - `_mm256_permute_ps` for intra-lane element permutation
/// - `_mm256_permute2f128_ps` for inter-lane 128-bit block permutation
///
/// Both operations use compile-time constant masks for optimal performance.
impl SimdShuffle for F32x8 {
    type Output = F32x8;

    /// Permutes elements within each 128-bit lane using AVX2 `_mm256_permute_ps`.
    ///
    /// This operation rearranges the 4 elements within each 128-bit half of the
    /// 256-bit vector independently using a compile-time mask. Each lane is
    /// permuted using the same pattern.
    ///
    /// # Performance
    ///
    /// - **Instruction**: Single `_mm256_permute_ps` AVX2 instruction
    /// - **Latency**: 1 cycle on most modern processors
    /// - **Throughput**: 2 operations per cycle (dual-port execution)
    /// - **Ports**: Executes on port 5 (Intel) or equivalent shuffle ports (AMD)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdly::simd::avx2::f32x8::F32x8;
    /// use simdly::simd::SimdShuffle;
    ///
    /// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
    ///
    /// // Broadcast element 0 in each lane: [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
    /// let broadcast = vec.permute::<0x00>();
    ///
    /// // Reverse elements in each lane: [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]
    /// let reversed = vec.permute::<0x1B>();
    /// ```
    #[inline(always)]
    fn permute<const MASK: i32>(&self) -> Self::Output {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_permute_ps(self.elements, MASK) },
        }
    }

    /// Permutes 128-bit lanes using AVX2 `_mm256_permute2f128_ps`.
    ///
    /// This operation rearranges the two 128-bit halves of the 256-bit vector,
    /// allowing duplication or swapping of the upper and lower 4-element groups.
    /// Essential for algorithms that need cross-lane data movement.
    ///
    /// # Performance
    ///
    /// - **Instruction**: Single `_mm256_permute2f128_ps` AVX2 instruction
    /// - **Latency**: 1-3 cycles depending on mask complexity  
    /// - **Throughput**: 0.5-1 operations per cycle (may use multiple ports)
    /// - **Domain crossing**: Some patterns may incur slight penalties
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdly::simd::avx2::f32x8::F32x8;
    /// use simdly::simd::SimdShuffle;
    ///
    /// let vec = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]);
    ///
    /// // Duplicate lower 4 elements: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    /// let low_dup = vec.permute2f128::<0x00>();
    ///
    /// // Swap halves: [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    /// let swapped = vec.permute2f128::<0x01>();
    /// ```
    #[inline(always)]
    fn permute2f128<const MASK: i32>(&self) -> Self::Output {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_permute2f128_ps(self.elements, self.elements, MASK) },
        }
    }
}

/// Implementation of element-wise addition for F32x8 vectors.
///
/// This implementation provides vectorized addition using AVX2 instructions,
/// performing 8 single-precision floating-point additions simultaneously.
///
/// # Performance
///
/// - **Vectorization**: Executes 8 additions in a single AVX2 instruction
/// - **Throughput**: ~8x faster than scalar addition for compatible workloads  
/// - **Latency**: Single-cycle execution on modern CPUs with sufficient execution units
/// - **Pipeline**: Fully pipelined operation allowing multiple additions per cycle
///
/// # Requirements
///
/// Both operands must have the same `size` (number of valid elements).
/// This ensures consistent behavior when working with partial vectors.
///
/// # Examples
///
/// ```rust
/// # use simdly::simd::avx2::f32x8::F32x8;
/// # use simdly::simd::SimdLoad;
/// # #[cfg(target_feature = "avx2")]
/// # {
/// let a = F32x8::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let b = F32x8::from(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
/// let result = a + b; // [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
/// # }
/// ```
///
/// # Panics
///
/// Panics in debug builds if the operands have different sizes.
impl Add for F32x8 {
    type Output = Self;

    /// Performs element-wise addition of two F32x8 vectors.
    ///
    /// Uses the AVX2 `_mm256_add_ps` intrinsic to add corresponding elements
    /// from both vectors simultaneously.
    ///
    /// # Arguments
    ///
    /// * `self` - Left operand vector
    /// * `rhs` - Right operand vector (must have same size as `self`)
    ///
    /// # Returns
    ///
    /// A new F32x8 vector containing the element-wise sum
    ///
    /// # Safety
    ///
    /// This function is safe as it only operates on the vector data and
    /// validates that both operands have compatible sizes.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the add function to perform element-wise addition
        Self {
            size: self.size,
            elements: unsafe { _mm256_add_ps(self.elements, rhs.elements) },
        }
    }
}

/// Implementation of element-wise subtraction for F32x8 vectors.
///
/// This implementation provides vectorized subtraction using AVX2 instructions,
/// performing 8 single-precision floating-point subtractions simultaneously.
impl Sub for F32x8 {
    type Output = Self;

    /// Performs element-wise subtraction of two F32x8 vectors.
    ///
    /// Uses the AVX2 `_mm256_sub_ps` intrinsic to subtract corresponding elements
    /// from both vectors simultaneously.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mm256_sub_ps(self.elements, rhs.elements) },
        }
    }
}

/// Implementation of element-wise multiplication for F32x8 vectors.
///
/// This implementation provides vectorized multiplication using AVX2 instructions,
/// performing 8 single-precision floating-point multiplications simultaneously.
impl Mul for F32x8 {
    type Output = Self;

    /// Performs element-wise multiplication of two F32x8 vectors.
    ///
    /// Uses the AVX2 `_mm256_mul_ps` intrinsic to multiply corresponding elements
    /// from both vectors simultaneously.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mm256_mul_ps(self.elements, rhs.elements) },
        }
    }
}

/// Implementation of element-wise division for F32x8 vectors.
///
/// This implementation provides vectorized division using AVX2 instructions,
/// performing 8 single-precision floating-point divisions simultaneously.
impl Div for F32x8 {
    type Output = Self;

    /// Performs element-wise division of two F32x8 vectors.
    ///
    /// Uses the AVX2 `_mm256_div_ps` intrinsic to divide corresponding elements
    /// from both vectors simultaneously.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mm256_div_ps(self.elements, rhs.elements) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    /// Helper function to create aligned memory for testing
    #[inline(always)]
    fn alloc_aligned(size: usize, align: usize) -> *mut f32 {
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), align).unwrap();
        unsafe { alloc(layout) as *mut f32 }
    }

    /// Helper function to deallocate aligned memory for testing
    #[inline(always)]
    fn dealloc_aligned(ptr: *mut f32, size: usize, align: usize) {
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), align).unwrap();
        unsafe { dealloc(ptr as *mut u8, layout) };
    }

    /// Helper function to extract vector elements for comparison in tests
    #[inline(always)]
    fn extract_elements(vec: &F32x8) -> [f32; 8] {
        let mut result = [0.0f32; 8];
        unsafe {
            std::arch::x86_64::_mm256_storeu_ps(result.as_mut_ptr(), vec.elements);
        }
        result
    }

    mod alignment_tests {
        use super::*;

        #[test]
        fn test_is_aligned_32_byte_boundary() {
            let aligned_ptr = alloc_aligned(8, 32);
            assert!(F32x8::is_aligned(aligned_ptr));
            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_is_not_aligned() {
            let data = [1.0f32; 16];
            let unaligned_ptr = unsafe { data.as_ptr().add(1) }; // Offset by 1 element (4 bytes)
            assert!(!F32x8::is_aligned(unaligned_ptr));
        }

        #[test]
        fn test_stack_array_alignment() {
            // Stack arrays may or may not be aligned depending on platform
            let data = [1.0f32; 8];
            let is_aligned = F32x8::is_aligned(data.as_ptr());
            // We don't assert a specific result since stack alignment varies
            // but the function should not panic
            let _ = is_aligned; // Suppress unused variable warning
        }
    }

    mod simd_load_tests {
        use super::*;

        #[test]
        fn test_from_slice_full() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(data.as_slice());

            assert_eq!(vec.size, 8);
            let elements = extract_elements(&vec);
            assert_eq!(elements, data);
        }

        #[test]
        fn test_from_slice_oversized() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let vec = F32x8::from(data.as_slice());

            assert_eq!(vec.size, 8); // Should still be 8, not 10
            let elements = extract_elements(&vec);
            assert_eq!(elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }

        #[test]
        fn test_from_slice_partial() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let vec = F32x8::from(data.as_slice());

            assert_eq!(vec.size, 5);
            let elements = extract_elements(&vec);
            // First 5 elements should match, rest are undefined but we don't test them
            assert_eq!(&elements[..5], &data);
        }

        #[test]
        fn test_load_aligned() {
            let aligned_ptr = alloc_aligned(8, 32);
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            unsafe {
                std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
            }

            let vec = unsafe { F32x8::load_aligned(aligned_ptr) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_load_unaligned() {
            let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let unaligned_ptr = unsafe { data.as_ptr().add(1) }; // Skip first element

            let vec = unsafe { F32x8::load_unaligned(unaligned_ptr) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }

        #[test]
        fn test_load_with_aligned_detection() {
            let aligned_ptr = alloc_aligned(8, 32);
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            unsafe {
                std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
            }

            let vec = unsafe { F32x8::load(aligned_ptr, 8) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_load_partial_single_element() {
            let data = [42.0f32];
            let vec = unsafe { F32x8::load_partial(data.as_ptr(), 1) };

            assert_eq!(vec.size, 1);
            let elements = extract_elements(&vec);
            assert_eq!(elements[0], 42.0);
        }

        #[test]
        fn test_load_partial_multiple_elements() {
            for size in 1..8 {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let vec = unsafe { F32x8::load_partial(data.as_ptr(), size) };

                assert_eq!(vec.size, size);
                let elements = extract_elements(&vec);

                // for i in 0..size {
                for (i, e) in elements.iter().enumerate().take(size) {
                    assert_eq!(*e, i as f32, "Mismatch at index {i} for size {size}");
                }
            }
        }

        #[test]
        fn test_load_partial_seven_elements() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let vec = unsafe { F32x8::load_partial(data.as_ptr(), 7) };

            assert_eq!(vec.size, 7);
            let elements = extract_elements(&vec);
            assert_eq!(&elements[..7], &data);
        }
    }

    mod simd_store_tests {
        use super::*;

        #[test]
        fn test_store_aligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(test_data.as_slice());

            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.store_aligned_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_store_unaligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(test_data.as_slice());

            let mut buffer = [0.0f32; 10];
            let unaligned_ptr = unsafe { buffer.as_mut_ptr().add(1) };

            unsafe { vec.store_unaligned_at(unaligned_ptr) };

            assert_eq!(&buffer[1..9], &test_data);
            assert_eq!(buffer[0], 0.0); // Should be unchanged
            assert_eq!(buffer[9], 0.0); // Should be unchanged
        }

        #[test]
        fn test_stream_at_aligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(test_data.as_slice());

            // Use properly aligned memory for streaming store
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_stream_at_alignment_check() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(test_data.as_slice());

            // Test with properly aligned memory
            let aligned_ptr = alloc_aligned(8, 32);
            assert!(
                F32x8::is_aligned(aligned_ptr),
                "Allocated memory should be aligned"
            );

            unsafe { vec.stream_at(aligned_ptr) };
            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);

            // Test alignment detection with unaligned memory
            let mut buffer = [0.0f32; 10];
            let unaligned_ptr = unsafe { buffer.as_mut_ptr().add(1) }; // Offset by 4 bytes
            assert!(
                !F32x8::is_aligned(unaligned_ptr),
                "Offset pointer should not be aligned"
            );
        }

        #[test]
        fn test_store_partial_single_element() {
            let test_data = [42.0];
            let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), 1) };

            let mut buffer = [0.0f32; 8];
            unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

            assert_eq!(buffer[0], 42.0);
            // Rest should remain zero
            for (i, e) in buffer.iter().enumerate().skip(1) {
                assert_eq!(*e, 0.0, "Non-zero value at index {i}");
            }
        }

        #[test]
        fn test_store_partial_multiple_elements() {
            for size in 1..8 {
                let test_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
                let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), size) };

                let mut buffer = [0.0f32; 8];
                unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

                // Check stored elements
                for (i, e) in buffer.iter().enumerate().take(size) {
                    assert_eq!(*e, (i + 1) as f32, "Mismatch at index {i} for size {size}");
                }

                // Check remaining elements are zero
                for (i, e) in buffer.iter().enumerate().skip(size) {
                    assert_eq!(*e, 0.0, "Non-zero value at index {i} for size {size}");
                }
            }
        }

        #[test]
        fn test_store_partial_seven_elements() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), 7) };

            let mut buffer = [0.0f32; 8];
            unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

            assert_eq!(&buffer[..7], &test_data);
            assert_eq!(buffer[7], 0.0);
        }
    }

    mod roundtrip_tests {
        use super::*;

        #[test]
        fn test_load_store_roundtrip_full() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(original.as_slice());

            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, original);
        }

        #[test]
        fn test_load_store_roundtrip_partial() {
            for size in 1..8 {
                let original: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
                let vec = unsafe { F32x8::load_partial(original.as_ptr(), size) };

                let mut result = [0.0f32; 8];
                unsafe { vec.store_at_partial(result.as_mut_ptr()) };

                assert_eq!(
                    &result[..size],
                    &original[..],
                    "Roundtrip failed for size {size}"
                );
            }
        }

        #[test]
        fn test_aligned_roundtrip() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let aligned_src = alloc_aligned(8, 32);
            let aligned_dst = alloc_aligned(8, 32);

            unsafe {
                std::ptr::copy_nonoverlapping(original.as_ptr(), aligned_src, 8);
            }

            let vec = unsafe { F32x8::load_aligned(aligned_src) };
            unsafe { vec.store_aligned_at(aligned_dst) };

            let result = unsafe { std::slice::from_raw_parts(aligned_dst, 8) };
            assert_eq!(result, &original);

            dealloc_aligned(aligned_src, 8, 32);
            dealloc_aligned(aligned_dst, 8, 32);
        }

        #[test]
        fn test_stream_roundtrip() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from(original.as_slice());

            // Stream to aligned memory
            let aligned_dst = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_dst) };

            let result = unsafe { std::slice::from_raw_parts(aligned_dst, 8) };
            assert_eq!(result, &original);

            dealloc_aligned(aligned_dst, 8, 32);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_zero_values() {
            let zeros = [0.0f32; 8];
            let vec = F32x8::from(zeros.as_slice());

            let mut result = [1.0f32; 8]; // Initialize with non-zero
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, zeros);
        }

        #[test]
        fn test_negative_values() {
            let negatives = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
            let vec = F32x8::from(negatives.as_slice());

            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, negatives);
        }

        #[test]
        fn test_special_float_values() {
            let special = [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                0.0,
                -0.0,
                f32::MIN,
                f32::MAX,
                f32::EPSILON,
            ];

            let vec = F32x8::from(special.as_slice());
            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result[0], f32::INFINITY);
            assert_eq!(result[1], f32::NEG_INFINITY);
            assert!(result[2].is_nan());
            assert_eq!(result[3], 0.0);
            assert_eq!(result[4], -0.0);
            assert_eq!(result[5], f32::MIN);
            assert_eq!(result[6], f32::MAX);
            assert_eq!(result[7], f32::EPSILON);
        }

        #[test]
        fn test_very_small_partial_load() {
            let single = [std::f32::consts::PI];
            let vec = unsafe { F32x8::load_partial(single.as_ptr(), 1) };

            assert_eq!(vec.size, 1);
            let elements = extract_elements(&vec);
            assert_eq!(elements[0], std::f32::consts::PI);
        }

        #[test]
        fn test_mixed_aligned_unaligned_operations() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            // Load unaligned
            let vec = unsafe { F32x8::load_unaligned(data.as_ptr()) };

            // Store aligned
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.store_aligned_at(aligned_ptr) };

            // Verify
            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_stream_with_special_values() {
            let special = [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                0.0,
                -0.0,
                f32::MIN,
                f32::MAX,
                f32::EPSILON,
            ];

            let vec = F32x8::from(special.as_slice());

            // Stream to aligned memory
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result[0], f32::INFINITY);
            assert_eq!(result[1], f32::NEG_INFINITY);
            assert!(result[2].is_nan());
            assert_eq!(result[3], 0.0);
            assert_eq!(result[4], -0.0);
            assert_eq!(result[5], f32::MIN);
            assert_eq!(result[6], f32::MAX);
            assert_eq!(result[7], f32::EPSILON);

            dealloc_aligned(aligned_ptr, 8, 32);
        }
    }

    #[cfg(debug_assertions)]
    mod debug_assertion_tests {
        use super::*;

        #[test]
        #[should_panic(expected = "data pointer can't be NULL")]
        fn test_from_slice_empty_panic() {
            let empty: &[f32] = &[];
            let _ = F32x8::from(empty);
        }

        #[test]
        #[should_panic(expected = "Size must be == 8")]
        fn test_load_wrong_size_panic() {
            let data = [1.0f32; 4];
            unsafe { F32x8::load(data.as_ptr(), 4) }; // Wrong size
        }

        #[test]
        #[should_panic(expected = "Size must be < 8")]
        fn test_load_partial_full_size_panic() {
            let data = [1.0f32; 8];
            unsafe { F32x8::load_partial(data.as_ptr(), 8) }; // Should be < 8
        }
    }
}
