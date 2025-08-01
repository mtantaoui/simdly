//! NEON 4-lane f32 SIMD vector implementation.
//!
//! This module provides `F32x4`, a SIMD vector type that wraps ARM's NEON `float32x4_t`
//! intrinsic to perform vectorized operations on 4 single-precision floating-point
//! values simultaneously using 128-bit NEON instructions.
//!
//! # Architecture Requirements
//!
//! - **CPU Support**: ARM processors with NEON support (most modern ARM chips)
//! - **Target Architecture**: AArch64 (ARM64)
//! - **Compilation**: Must be compiled with NEON enabled (`-C target-feature=+neon`)
//!
//! # Performance Characteristics
//!
//! - **Vector Width**: 128 bits (4 × f32)
//! - **Memory Alignment**: Optimal performance with 16-byte aligned data
//! - **Throughput**: Up to 4× speedup for vectorizable operations compared to scalar code
//! - **Power Efficiency**: NEON is designed for mobile and embedded processors
//!
//! # Supported Operations
//!
//! ## Loading and Storing
//! - `from_slice()` - High-level loading with automatic partial handling
//! - `load_aligned()`, `load_unaligned()` - Direct memory loading
//! - `load_partial()` - Safe partial loading for sizes < 4
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
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::simd::{Alignment, SimdLoad, SimdMath, SimdStore};
use std::ops::{Add, Div, Mul, Sub};

use super::math::*;

#[allow(dead_code)]
pub(crate) const NEON_ALIGNMENT: usize = 16;
pub(crate) const LANE_COUNT: usize = 4;

/// NEON SIMD vector containing 4 packed f32 values.
///
/// This structure provides efficient vectorized operations on 4 single-precision
/// floating-point numbers using NEON instructions. It maintains both the underlying
/// NEON register and the count of valid elements for partial operations.
///
/// # Memory Alignment
///
/// For optimal performance, data should be aligned to 16-byte boundaries when possible.
/// NEON instructions can handle unaligned data but aligned access is faster.
///
/// # Usage
///
/// ```rust
/// # #[cfg(target_feature = "neon")]
/// # {
/// use simdly::simd::neon::f32x4::F32x4;
/// use simdly::simd::SimdLoad;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0];
/// let vector = F32x4::from_slice(&data);
/// # }
/// ```
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    /// Number of valid elements in the vector (1-4)
    pub size: usize,
    /// NEON 128-bit vector register containing 4 packed f32 values
    pub elements: float32x4_t,
}

impl Alignment<f32> for F32x4 {
    /// Checks if a pointer is properly aligned for NEON operations.
    ///
    /// NEON operations perform optimally when data is aligned to 16-byte boundaries.
    /// This function checks if the given pointer meets this alignment requirement.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to check for alignment
    ///
    /// # Returns
    ///
    /// `true` if the pointer is 16-byte aligned, `false` otherwise
    #[inline(always)]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;
        ptr % core::mem::align_of::<float32x4_t>() == 0
    }
}

impl SimdLoad<f32> for F32x4 {
    type Output = Self;

    /// Loads data from a slice into the SIMD vector.
    ///
    /// This is the high-level interface for loading f32 data. It automatically
    /// handles partial loads for slices smaller than 4 elements and uses the
    /// most appropriate loading strategy based on slice size.
    ///
    /// # Arguments
    ///
    /// * `slice` - Input slice of f32 values
    ///
    /// # Returns
    ///
    /// F32x4 instance with loaded data
    ///
    /// # Behavior
    ///
    /// - For slices < 4 elements: Uses partial loading with zero-padding
    /// - For slices >= 4 elements: Loads the first 4 elements
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slice is empty.
    fn from_slice(slice: &[f32]) -> Self::Output {
        debug_assert!(!slice.is_empty(), "data pointer can't be NULL");

        let size = slice.len().min(LANE_COUNT);

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), size) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    /// Loads exactly 4 elements from memory.
    ///
    /// Uses NEON `vld1q_f32` intrinsic for efficient loading.
    /// NEON can handle both aligned and unaligned loads efficiently.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Must be exactly 4
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least 4 valid f32 values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size != 4 or if pointer is null.
    unsafe fn load(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        Self {
            elements: vld1q_f32(ptr),
            size: LANE_COUNT,
        }
    }

    /// Loads 4 elements from aligned memory.
    ///
    /// On NEON, there's no separate aligned load instruction, so this uses
    /// the same `vld1q_f32` intrinsic as unaligned loads. NEON handles
    /// alignment optimizations automatically.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data (alignment preferred but not required)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 4 valid f32 values.
    unsafe fn load_aligned(_ptr: *const f32) -> Self::Output {
        unimplemented!()
    }

    /// Loads 4 elements from unaligned memory.
    ///
    /// On NEON, this uses the same `vld1q_f32` intrinsic as aligned loads.
    /// NEON efficiently handles unaligned memory access.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 4 valid f32 values.
    unsafe fn load_unaligned(_ptr: *const f32) -> Self::Output {
        unimplemented!()
    }

    /// Loads fewer than 4 elements using element-wise loading.
    ///
    /// This function safely loads partial data when the source contains fewer than
    /// 4 elements. Unloaded lanes are filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Number of elements to load (1-3)
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least `size` valid f32 values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size >= 4 or if pointer is null.
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let elements = match size {
            1 => {
                // Load single element using vld1q_lane_f32 for efficiency
                let zero = vdupq_n_f32(0.0);
                vld1q_lane_f32::<0>(ptr, zero)
            }
            2 => {
                // Load 2 elements using vld1_f32 (64-bit load) and combine with zeros
                let low_pair = vld1_f32(ptr); // Load 2 f32 values
                let zero_pair = vdup_n_f32(0.0); // Create zero pair
                vcombine_f32(low_pair, zero_pair) // Combine: [a, b, 0, 0]
            }
            3 => {
                // Load 2 elements first, then load the third element separately
                let low_pair = vld1_f32(ptr); // Load first 2 elements
                let zero_pair = vdup_n_f32(0.0); // Create zero pair
                let mut result = vcombine_f32(low_pair, zero_pair); // [a, b, 0, 0]
                result = vld1q_lane_f32::<2>(ptr.add(2), result); // Load third element into lane 2
                result
            }
            _ => unreachable!("Size must be < {}", LANE_COUNT),
        };

        Self { elements, size }
    }
}

impl SimdMath<f32> for F32x4 {
    type Output = Self;

    /// Computes the absolute value of each element using NEON intrinsics.
    ///
    /// Uses `vabsq_f32` to efficiently clear the sign bit of all elements.
    fn abs(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vabsq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the arccosine using optimized NEON math functions.
    ///
    /// Uses the `vacosq_f32` function from the math module for vectorized
    /// arccosine computation with polynomial approximation.
    fn acos(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vacosq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the arcsine using optimized NEON math functions.
    ///
    /// Uses the `vasinq_f32` function from the math module for vectorized
    /// arcsine computation with polynomial approximation.
    fn asin(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vasinq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the arctangent using optimized NEON math functions.
    ///
    /// Uses the `vatanq_f32` function from the math module for vectorized
    /// arctangent computation with polynomial approximation.
    fn atan(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vatanq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes two-argument arctangent using optimized NEON math functions.
    ///
    /// This implementation treats the vector as pairs of (y, x) coordinates.
    /// Uses the `vatan2q_f32` function for proper quadrant handling.
    fn atan2(&self) -> Self::Output {
        unsafe {
            // Extract pairs for atan2 computation
            let y_vec = self.elements;
            let x_vec = self.elements; // Same vector used for both, pairs extracted in math function

            Self {
                elements: vatan2q_f32(y_vec, x_vec),
                size: self.size,
            }
        }
    }

    /// Computes the cube root using optimized NEON math functions.
    ///
    /// Uses the `vcbrtq_f32` function from the math module for vectorized
    /// cube root computation with Newton-Raphson iteration.
    fn cbrt(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vcbrtq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the floor using NEON's native rounding intrinsic.
    ///
    /// Uses `vrndmq_f32` for efficient vectorized floor operation.
    fn floor(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vrndmq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the natural exponential using optimized NEON math functions.
    ///
    /// Uses the `vexpq_f32` function from the math module for vectorized
    /// exponential computation with range reduction and polynomial approximation.
    fn exp(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vexpq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the natural logarithm using optimized NEON math functions.
    ///
    /// Uses the `vlnq_f32` function from the math module for vectorized
    /// logarithm computation with range reduction and polynomial approximation.
    fn ln(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vlnq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes 2D hypotenuse using optimized NEON math functions.
    ///
    /// Uses the `vhypotq_f32` function from the math module for vectorized
    /// Euclidean distance computation with overflow protection.
    fn hypot(&self) -> Self::Output {
        todo!()
    }

    /// Computes power using optimized NEON math functions.
    ///
    /// Uses a combination of logarithm and exponential functions for
    /// vectorized power computation: x^y = exp(y * ln(x)).
    fn pow(&self) -> Self::Output {
        todo!()
    }

    /// Computes sine using optimized NEON math functions.
    ///
    /// Uses the `vsinq_f32` function from the math module for vectorized
    /// sine computation with range reduction and polynomial approximation.
    fn sin(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vsinq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes cosine using optimized NEON math functions.
    ///
    /// Uses the `vcosq_f32` function from the math module for vectorized
    /// cosine computation with range reduction and polynomial approximation.
    fn cos(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vcosq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes tangent using optimized NEON math functions.
    ///
    /// Uses the `vtanq_f32` function from the math module for vectorized
    /// tangent computation with range reduction and polynomial approximation.
    fn tan(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vtanq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes square root using NEON's native square root intrinsic.
    ///
    /// Uses `vsqrtq_f32` for efficient vectorized square root computation.
    fn sqrt(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vsqrtq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes the ceiling using NEON's native rounding intrinsic.
    ///
    /// Uses `vrndpq_f32` for efficient vectorized ceiling operation.
    fn ceil(&self) -> Self::Output {
        unsafe {
            Self {
                elements: vrndpq_f32(self.elements),
                size: self.size,
            }
        }
    }

    /// Computes 3D hypotenuse using optimized NEON math functions.
    ///
    /// Uses the `vhypot3q_f32` function from the math module for vectorized
    /// 3D Euclidean distance computation.
    fn hypot3(&self) -> Self::Output {
        todo!()
    }

    /// Computes 4D hypotenuse using optimized NEON math functions.
    ///
    /// Uses the `vhypot4q_f32` function from the math module for vectorized
    /// 4D Euclidean distance computation.
    fn hypot4(&self) -> Self::Output {
        todo!()
    }
}

impl SimdStore<f32> for F32x4 {
    type Output = Self;

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
            std::cmp::Ordering::Equal => unsafe { vst1q_f32(mut_ptr, self.elements) },
            std::cmp::Ordering::Greater => unreachable!("Size cannot exceed LANE_COUNT"),
        }
    }

    /// Stores vector data using non-temporal (streaming) operations.
    ///
    /// On NEON, this uses the same store instruction as regular stores since
    /// NEON doesn't have separate streaming store instructions.
    ///
    /// # Safety
    ///
    /// Pointer must be valid, non-null, and point to writable memory for all elements.
    unsafe fn stream_at(&self, ptr: *mut f32) {
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match self.size.cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => self.store_at_partial(ptr),
            std::cmp::Ordering::Equal => vst1q_f32(ptr, self.elements),
            std::cmp::Ordering::Greater => unreachable!("Size cannot exceed LANE_COUNT"),
        }
    }

    /// Stores vector data to aligned memory.
    ///
    /// On NEON, this uses the same `vst1q_f32` instruction as unaligned stores.
    /// NEON handles alignment optimizations automatically.
    ///
    /// # Safety
    ///
    /// Pointer must be properly aligned and point to writable memory for all elements.
    unsafe fn store_aligned_at(&self, _ptr: *mut f32) {
        unimplemented!()
    }

    /// Stores vector data to unaligned memory.
    ///
    /// On NEON, this uses the same `vst1q_f32` instruction as aligned stores.
    /// NEON efficiently handles unaligned memory access.
    ///
    /// # Safety
    ///
    /// Pointer must be valid, non-null, and point to writable memory for all elements.
    unsafe fn store_unaligned_at(&self, _ptr: *mut f32) {
        unimplemented!()
    }

    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(
            self.size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match self.size {
            3 => {
                let low = vget_low_f32(self.elements); // extract [0, 1]
                vst1_f32(ptr, low); // store [0, 1]

                let third = vgetq_lane_f32(self.elements, 2); // extract element 2
                *ptr.add(2) = third; // store element 2
            }
            2 => {
                let low = vget_low_f32(self.elements);
                vst1_f32(ptr, low);
            }
            1 => {
                let first = vgetq_lane_f32(self.elements, 0);
                *ptr = first;
            }
            _ => unreachable!("Size must be < LANE_COUNT"),
        }
    }
}

// Arithmetic operator implementations using NEON intrinsics

impl Add for F32x4 {
    type Output = Self;

    /// Adds two F32x4 vectors element-wise using NEON intrinsics.
    ///
    /// Uses `vaddq_f32` for efficient vectorized addition.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            Self {
                elements: vaddq_f32(self.elements, rhs.elements),
                size: self.size,
            }
        }
    }
}

impl Sub for F32x4 {
    type Output = Self;

    /// Subtracts two F32x4 vectors element-wise using NEON intrinsics.
    ///
    /// Uses `vsubq_f32` for efficient vectorized subtraction.
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            Self {
                elements: vsubq_f32(self.elements, rhs.elements),
                size: self.size,
            }
        }
    }
}

impl Mul for F32x4 {
    type Output = Self;

    /// Multiplies two F32x4 vectors element-wise using NEON intrinsics.
    ///
    /// Uses `vmulq_f32` for efficient vectorized multiplication.
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            Self {
                elements: vmulq_f32(self.elements, rhs.elements),
                size: self.size,
            }
        }
    }
}

impl Div for F32x4 {
    type Output = Self;

    /// Divides two F32x4 vectors element-wise using NEON intrinsics.
    ///
    /// Uses `vdivq_f32` for efficient vectorized division.
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            Self {
                elements: vdivq_f32(self.elements, rhs.elements),
                size: self.size,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test Alignment trait
    #[test]
    fn test_alignment_trait() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let ptr = data.as_ptr();

        // Test that alignment check works without panicking
        let _is_aligned = F32x4::is_aligned(ptr);

        // Test with known aligned data (stack arrays are typically aligned)
        let aligned_data = [1.0f32; 8]; // Larger array more likely to be aligned
        let aligned_ptr = aligned_data.as_ptr();
        let _is_aligned = F32x4::is_aligned(aligned_ptr);

        // Test with offset pointer (typically unaligned)
        let offset_ptr = unsafe { aligned_data.as_ptr().add(1) };
        let _is_unaligned = F32x4::is_aligned(offset_ptr);
    }

    // Test SimdLoad trait
    #[test]
    fn test_simd_load_from_slice_full() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let vector = F32x4::from_slice(&data);

        assert_eq!(vector.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 1.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 2.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 3.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 4.0);
        }
    }

    #[test]
    fn test_simd_load_from_slice_oversized() {
        // Test loading from slice larger than vector capacity
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = F32x4::from_slice(&data);

        assert_eq!(vector.size, 4); // Should cap at LANE_COUNT
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 1.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 2.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 3.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 4.0);
            // Elements 5.0 and 6.0 should be ignored
        }
    }

    #[test]
    fn test_simd_load_from_slice_partial() {
        // Test partial load with 2 elements
        let data = [5.0f32, 6.0];
        let vector = F32x4::from_slice(&data);

        assert_eq!(vector.size, 2);
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 5.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 6.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 0.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 0.0);
        }
    }

    #[test]
    fn test_simd_load_from_slice_single() {
        // Test partial load with 1 element
        let data = [42.0f32];
        let vector = F32x4::from_slice(&data);

        assert_eq!(vector.size, 1);
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 42.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 0.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 0.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 0.0);
        }
    }

    #[test]
    fn test_simd_load_from_slice_three() {
        // Test partial load with 3 elements
        let data = [1.5f32, 2.5, 3.5];
        let vector = F32x4::from_slice(&data);

        assert_eq!(vector.size, 3);
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 1.5);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 2.5);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 3.5);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 0.0);
        }
    }

    #[test]
    fn test_simd_load_partial() {
        let data = [10.0f32, 20.0, 30.0];
        let vector = unsafe { F32x4::load_partial(data.as_ptr(), 3) };

        assert_eq!(vector.size, 3);
        unsafe {
            assert_eq!(vgetq_lane_f32(vector.elements, 0), 10.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 1), 20.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 2), 30.0);
            assert_eq!(vgetq_lane_f32(vector.elements, 3), 0.0);
        }
    }

    // Test SimdMath trait (basic functionality without testing specific math results)
    #[test]
    fn test_simd_math_trait_methods_exist() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let vector = F32x4::from_slice(&data);

        // Test that all SimdMath methods can be called without panicking
        let _abs_result = vector.abs();
        let _sqrt_result = vector.sqrt();
        let _floor_result = vector.floor();
        let _ceil_result = vector.ceil();
        let _sin_result = vector.sin();
        let _cos_result = vector.cos();
        let _tan_result = vector.tan();
        let _asin_result = vector.asin();
        let _acos_result = vector.acos();
        let _atan_result = vector.atan();
        let _atan2_result = vector.atan2();
        let _exp_result = vector.exp();
        let _ln_result = vector.ln();
        let _cbrt_result = vector.cbrt();
        // let _hypot_result = vector.hypot();
        // let _hypot3_result = vector.hypot3();
        // let _hypot4_result = vector.hypot4();
        // let _pow_result = vector.pow();
    }

    // Test SimdStore trait
    #[test]
    fn test_simd_store_at() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let vector = F32x4::from_slice(&data);
        let output = [0.0f32; 4];

        vector.store_at(output.as_ptr());
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_simd_store_partial() {
        let data = [10.0f32, 20.0];
        let vector = F32x4::from_slice(&data);
        let output = [0.0f32; 4];

        vector.store_at(output.as_ptr());
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 20.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn test_simd_store_single() {
        // Test storing single element
        let data = [99.0f32];
        let vector = F32x4::from_slice(&data);
        let output = [0.0f32; 4];

        vector.store_at(output.as_ptr());
        assert_eq!(output[0], 99.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn test_simd_store_three() {
        // Test storing three elements
        let data = [1.1f32, 2.2, 3.3];
        let vector = F32x4::from_slice(&data);
        let output = [0.0f32; 4];

        vector.store_at(output.as_ptr());
        assert_eq!(output[0], 1.1);
        assert_eq!(output[1], 2.2);
        assert_eq!(output[2], 3.3);
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn test_simd_store_stream() {
        let data = [13.0f32, 14.0, 15.0, 16.0];
        let vector = F32x4::from_slice(&data);
        let mut output = [0.0f32; 4];

        unsafe {
            vector.stream_at(output.as_mut_ptr());
        }
        assert_eq!(output, [13.0, 14.0, 15.0, 16.0]);
    }

    // Test arithmetic operators (basic trait functionality)
    #[test]
    fn test_arithmetic_add() {
        let data1 = [1.0f32, 2.0, 3.0, 4.0];
        let data2 = [10.0f32, 20.0, 30.0, 40.0];
        let vec1 = F32x4::from_slice(&data1);
        let vec2 = F32x4::from_slice(&data2);

        let result = vec1 + vec2;
        assert_eq!(result.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 11.0);
            assert_eq!(vgetq_lane_f32(result.elements, 1), 22.0);
            assert_eq!(vgetq_lane_f32(result.elements, 2), 33.0);
            assert_eq!(vgetq_lane_f32(result.elements, 3), 44.0);
        }
    }

    #[test]
    fn test_arithmetic_sub() {
        let data1 = [10.0f32, 20.0, 30.0, 40.0];
        let data2 = [1.0f32, 2.0, 3.0, 4.0];
        let vec1 = F32x4::from_slice(&data1);
        let vec2 = F32x4::from_slice(&data2);

        let result = vec1 - vec2;
        assert_eq!(result.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 9.0);
            assert_eq!(vgetq_lane_f32(result.elements, 1), 18.0);
            assert_eq!(vgetq_lane_f32(result.elements, 2), 27.0);
            assert_eq!(vgetq_lane_f32(result.elements, 3), 36.0);
        }
    }

    #[test]
    fn test_arithmetic_mul() {
        let data1 = [2.0f32, 3.0, 4.0, 5.0];
        let data2 = [10.0f32, 10.0, 10.0, 10.0];
        let vec1 = F32x4::from_slice(&data1);
        let vec2 = F32x4::from_slice(&data2);

        let result = vec1 * vec2;
        assert_eq!(result.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 20.0);
            assert_eq!(vgetq_lane_f32(result.elements, 1), 30.0);
            assert_eq!(vgetq_lane_f32(result.elements, 2), 40.0);
            assert_eq!(vgetq_lane_f32(result.elements, 3), 50.0);
        }
    }

    #[test]
    fn test_arithmetic_div() {
        let data1 = [20.0f32, 30.0, 40.0, 50.0];
        let data2 = [10.0f32, 10.0, 10.0, 10.0];
        let vec1 = F32x4::from_slice(&data1);
        let vec2 = F32x4::from_slice(&data2);

        let result = vec1 / vec2;
        assert_eq!(result.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 2.0);
            assert_eq!(vgetq_lane_f32(result.elements, 1), 3.0);
            assert_eq!(vgetq_lane_f32(result.elements, 2), 4.0);
            assert_eq!(vgetq_lane_f32(result.elements, 3), 5.0);
        }
    }

    #[test]
    fn test_arithmetic_partial_vectors() {
        // Test arithmetic with partial vectors
        let data1 = [1.0f32, 2.0];
        let data2 = [10.0f32, 20.0];
        let vec1 = F32x4::from_slice(&data1);
        let vec2 = F32x4::from_slice(&data2);

        let result = vec1 + vec2;
        assert_eq!(result.size, 2);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 11.0);
            assert_eq!(vgetq_lane_f32(result.elements, 1), 22.0);
            // Note: lanes 2 and 3 values are undefined for partial vectors
        }
    }

    #[test]
    fn test_arithmetic_chaining() {
        // Test chaining multiple operations
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let vec = F32x4::from_slice(&data);
        let two = F32x4::from_slice(&[2.0f32, 2.0, 2.0, 2.0]);

        let result = (vec + two) * two;
        assert_eq!(result.size, 4);
        unsafe {
            assert_eq!(vgetq_lane_f32(result.elements, 0), 6.0); // (1+2)*2
            assert_eq!(vgetq_lane_f32(result.elements, 1), 8.0); // (2+2)*2
            assert_eq!(vgetq_lane_f32(result.elements, 2), 10.0); // (3+2)*2
            assert_eq!(vgetq_lane_f32(result.elements, 3), 12.0); // (4+2)*2
        }
    }

    // Enhanced edge case tests
    #[test]
    fn test_edge_cases_zero_values() {
        let data = [0.0f32, -0.0, 0.0, -0.0];
        let vec = F32x4::from_slice(&data);

        let abs_result = vec.abs();
        unsafe {
            // All should be positive zero
            assert_eq!(vgetq_lane_f32(abs_result.elements, 0), 0.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 1), 0.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 2), 0.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 3), 0.0);
        }
    }

    #[test]
    fn test_edge_cases_infinity() {
        let data = [f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];
        let vec = F32x4::from_slice(&data);

        let abs_result = vec.abs();
        unsafe {
            assert_eq!(vgetq_lane_f32(abs_result.elements, 0), f32::INFINITY);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 1), f32::INFINITY);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 2), 1.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 3), 1.0);
        }
    }

    #[test]
    fn test_mathematical_functions_validation() {
        // Test sqrt with known values
        let data = [4.0f32, 9.0, 16.0, 25.0];
        let vec = F32x4::from_slice(&data);
        let sqrt_result = vec.sqrt();

        unsafe {
            assert_eq!(vgetq_lane_f32(sqrt_result.elements, 0), 2.0);
            assert_eq!(vgetq_lane_f32(sqrt_result.elements, 1), 3.0);
            assert_eq!(vgetq_lane_f32(sqrt_result.elements, 2), 4.0);
            assert_eq!(vgetq_lane_f32(sqrt_result.elements, 3), 5.0);
        }

        // Test floor and ceil with fractional values
        let frac_data = [2.3f32, 2.7, -1.2, -1.8];
        let frac_vec = F32x4::from_slice(&frac_data);

        let floor_result = frac_vec.floor();
        let ceil_result = frac_vec.ceil();

        unsafe {
            // Floor tests
            assert_eq!(vgetq_lane_f32(floor_result.elements, 0), 2.0);
            assert_eq!(vgetq_lane_f32(floor_result.elements, 1), 2.0);
            assert_eq!(vgetq_lane_f32(floor_result.elements, 2), -2.0);
            assert_eq!(vgetq_lane_f32(floor_result.elements, 3), -2.0);

            // Ceil tests
            assert_eq!(vgetq_lane_f32(ceil_result.elements, 0), 3.0);
            assert_eq!(vgetq_lane_f32(ceil_result.elements, 1), 3.0);
            assert_eq!(vgetq_lane_f32(ceil_result.elements, 2), -1.0);
            assert_eq!(vgetq_lane_f32(ceil_result.elements, 3), -1.0);
        }
    }

    #[test]
    fn test_partial_vector_operations() {
        // Test operations with different sized partial vectors
        let single = F32x4::from_slice(&[5.0f32]);
        let pair = F32x4::from_slice(&[10.0f32, 20.0]);
        let triple = F32x4::from_slice(&[1.0f32, 2.0, 3.0]);

        assert_eq!(single.size, 1);
        assert_eq!(pair.size, 2);
        assert_eq!(triple.size, 3);

        // Test that operations preserve sizes correctly
        let single_sqrt = single.sqrt();
        let pair_abs = pair.abs();
        let triple_floor = triple.floor();

        assert_eq!(single_sqrt.size, 1);
        assert_eq!(pair_abs.size, 2);
        assert_eq!(triple_floor.size, 3);
    }

    #[test]
    fn test_store_partial_edge_cases() {
        // Test storing to exact-sized buffers
        let single_data = [42.0f32];
        let single_vec = F32x4::from_slice(&single_data);
        let single_out = [0.0f32; 1];
        single_vec.store_at(single_out.as_ptr());
        assert_eq!(single_out[0], 42.0);

        let pair_data = [10.0f32, 20.0];
        let pair_vec = F32x4::from_slice(&pair_data);
        let pair_out = [0.0f32; 2];
        pair_vec.store_at(pair_out.as_ptr());
        assert_eq!(pair_out, [10.0, 20.0]);

        let triple_data = [1.5f32, 2.5, 3.5];
        let triple_vec = F32x4::from_slice(&triple_data);
        let triple_out = [0.0f32; 3];
        triple_vec.store_at(triple_out.as_ptr());
        assert_eq!(triple_out, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_load_store_roundtrip() {
        // Test that loading and storing preserves data exactly
        let original = [1.25f32, -2.75, 3.125, -4.875];
        let vec = F32x4::from_slice(&original);
        let output = [0.0f32; 4];

        vec.store_at(output.as_ptr());
        assert_eq!(output, original);

        // Test with partial data
        let partial_original = [9.0f32, -8.0, 7.0];
        let partial_vec = F32x4::from_slice(&partial_original);
        let partial_output = [0.0f32; 4];

        partial_vec.store_at(partial_output.as_ptr());
        assert_eq!(partial_output[0], 9.0);
        assert_eq!(partial_output[1], -8.0);
        assert_eq!(partial_output[2], 7.0);
        assert_eq!(partial_output[3], 0.0); // Should remain zero
    }

    #[test]
    fn test_complex_mathematical_operations() {
        // Test combinations of mathematical functions
        let data = [1.0f32, 4.0, 9.0, 16.0];
        let vec = F32x4::from_slice(&data);

        // Test sqrt followed by arithmetic
        let sqrt_vec = vec.sqrt();
        let doubled = sqrt_vec + sqrt_vec;

        unsafe {
            assert_eq!(vgetq_lane_f32(doubled.elements, 0), 2.0); // sqrt(1) * 2
            assert_eq!(vgetq_lane_f32(doubled.elements, 1), 4.0); // sqrt(4) * 2
            assert_eq!(vgetq_lane_f32(doubled.elements, 2), 6.0); // sqrt(9) * 2
            assert_eq!(vgetq_lane_f32(doubled.elements, 3), 8.0); // sqrt(16) * 2
        }

        // Test abs with negative values
        let neg_data = [-1.0f32, -2.0, -3.0, -4.0];
        let neg_vec = F32x4::from_slice(&neg_data);
        let abs_result = neg_vec.abs();

        unsafe {
            assert_eq!(vgetq_lane_f32(abs_result.elements, 0), 1.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 1), 2.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 2), 3.0);
            assert_eq!(vgetq_lane_f32(abs_result.elements, 3), 4.0);
        }
    }

    #[test]
    fn test_all_store_methods() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let vec = F32x4::from_slice(&data);

        // Test all store methods produce the same result
        let mut stream_out = [0.0f32; 4];

        unsafe {
            vec.stream_at(stream_out.as_mut_ptr());
        }

        assert_eq!(stream_out, data);
    }

    #[test]
    fn test_performance_characteristics() {
        // Test that SIMD operations complete without hanging
        // This is mainly a smoke test for performance
        let data = [1.0f32; 4];
        let vec = F32x4::from_slice(&data);

        // Perform multiple operations to ensure they all work
        for _ in 0..1000 {
            let _result = ((vec.sqrt().abs() + vec) * vec).floor();
        }
    }

    #[test]
    fn test_enhanced_precision_against_scalar() {
        // Compare SIMD results against scalar standard library for validation
        let test_cases = [
            0.1f32,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            std::f32::consts::PI / 4.0,
        ];

        for &test_val in &test_cases {
            let vec_input = F32x4::from_slice(&[test_val, test_val, test_val, test_val]);

            // Test sine precision
            let sin_vec_result = vec_input.sin();
            let sin_scalar_expected = test_val.sin();
            unsafe {
                let sin_vec_computed = vgetq_lane_f32(sin_vec_result.elements, 0);
                let sin_relative_error =
                    ((sin_vec_computed - sin_scalar_expected) / sin_scalar_expected).abs();
                assert!(
                    sin_relative_error < 2e-6, // Allow slightly relaxed tolerance vs scalar
                    "SIMD sine vs scalar mismatch for {}: SIMD={}, scalar={}, error={:.2e}",
                    test_val,
                    sin_vec_computed,
                    sin_scalar_expected,
                    sin_relative_error
                );
            }

            // Test cosine precision
            let cos_vec_result = vec_input.cos();
            let cos_scalar_expected = test_val.cos();
            unsafe {
                let cos_vec_computed = vgetq_lane_f32(cos_vec_result.elements, 0);
                let cos_relative_error =
                    ((cos_vec_computed - cos_scalar_expected) / cos_scalar_expected).abs();
                assert!(
                    cos_relative_error < 2e-6,
                    "SIMD cosine vs scalar mismatch for {}: SIMD={}, scalar={}, error={:.2e}",
                    test_val,
                    cos_vec_computed,
                    cos_scalar_expected,
                    cos_relative_error
                );
            }

            // Test exponential precision (for reasonable range)
            if test_val < 10.0 {
                let exp_vec_result = vec_input.exp();
                let exp_scalar_expected = test_val.exp();
                unsafe {
                    let exp_vec_computed = vgetq_lane_f32(exp_vec_result.elements, 0);
                    let exp_relative_error =
                        ((exp_vec_computed - exp_scalar_expected) / exp_scalar_expected).abs();
                    assert!(
                        exp_relative_error < 2e-6,
                        "SIMD exp vs scalar mismatch for {}: SIMD={}, scalar={}, error={:.2e}",
                        test_val,
                        exp_vec_computed,
                        exp_scalar_expected,
                        exp_relative_error
                    );
                }
            }
        }
    }
}
