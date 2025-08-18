//! Platform-agnostic scalar operations for slice data.
//!
//! This module provides scalar fallback implementations for vector operations
//! that work on any platform regardless of SIMD support. These functions serve as:
//!
//! - **Fallback implementations** when SIMD is not available
//! - **Reference implementations** for correctness testing
//! - **Performance baselines** for benchmarking SIMD optimizations
//! - **Compatibility layer** for platforms without SIMD support
//!
//! # Performance Characteristics
//!
//! - **Predictable**: Consistent performance across all platforms
//! - **Low overhead**: Minimal function call and setup costs
//! - **Cache friendly**: Sequential memory access patterns
//! - **Compiler optimized**: Leverages compiler auto-vectorization when possible
//!
//! # When to Use
//!
//! These functions are automatically used by the high-level traits when:
//! - Array size is below SIMD efficiency thresholds
//! - SIMD instructions are not available on the target CPU
//! - Debug builds where SIMD complexity isn't warranted

use crate::{simd::SimdMath, PARALLEL_SIMD_THRESHOLD};

#[cfg(all(neon, not(avx2)))]
use crate::simd::neon::slice::{
    parallel_simd_abs, parallel_simd_acos, parallel_simd_asin, parallel_simd_atan,
    parallel_simd_atan2, parallel_simd_cbrt, parallel_simd_ceil, parallel_simd_cos,
    parallel_simd_exp, parallel_simd_floor, parallel_simd_hypot, parallel_simd_hypot3,
    parallel_simd_hypot4, parallel_simd_ln, parallel_simd_pow, parallel_simd_sin,
    parallel_simd_sqrt, parallel_simd_tan, simd_abs, simd_acos, simd_asin, simd_atan, simd_atan2,
    simd_cbrt, simd_ceil, simd_cos, simd_exp, simd_floor, simd_hypot, simd_hypot3, simd_hypot4,
    simd_ln, simd_pow, simd_sin, simd_sqrt, simd_tan,
};

#[cfg(avx2)]
use crate::simd::avx2::slice::{
    parallel_simd_abs, parallel_simd_acos, parallel_simd_asin, parallel_simd_atan,
    parallel_simd_atan2, parallel_simd_cbrt, parallel_simd_ceil, parallel_simd_cos,
    parallel_simd_exp, parallel_simd_floor, parallel_simd_hypot, parallel_simd_hypot3,
    parallel_simd_hypot4, parallel_simd_ln, parallel_simd_pow, parallel_simd_sin,
    parallel_simd_sqrt, parallel_simd_tan, simd_abs, simd_acos, simd_asin, simd_atan, simd_atan2,
    simd_cbrt, simd_ceil, simd_cos, simd_exp, simd_floor, simd_hypot, simd_hypot3, simd_hypot4,
    simd_ln, simd_pow, simd_sin, simd_sqrt, simd_tan,
};

/// Implementation of mathematical operations for `Vec<f32>` using platform-optimized SIMD.
///
/// This implementation provides the same vectorized mathematical functions as the
/// slice implementation, but operates directly on owned vectors. It delegates to
/// the slice implementation for actual computation.
///
/// # Usage Pattern
///
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let results = data.cos(); // Uses platform-optimized SIMD cosine
/// ```
impl SimdMath for Vec<f32> {
    type Output = Vec<f32>;

    /// Computes absolute value using platform-optimized SIMD instructions.
    #[inline(always)]
    fn abs(&self) -> Self::Output {
        simd_abs(self)
    }

    /// Computes arccosine using platform-optimized SIMD instructions.
    #[inline(always)]
    fn acos(&self) -> Self::Output {
        simd_acos(self)
    }

    /// Computes arcsine using platform-optimized SIMD instructions.
    #[inline(always)]
    fn asin(&self) -> Self::Output {
        simd_asin(self)
    }

    /// Computes arctangent using platform-optimized SIMD instructions.
    #[inline(always)]
    fn atan(&self) -> Self::Output {
        simd_atan(self)
    }

    /// Computes two-argument arctangent using platform-optimized SIMD instructions.
    #[inline(always)]
    fn atan2(&self, other: Self) -> Self::Output {
        simd_atan2(self, &other)
    }

    /// Computes cube root using platform-optimized SIMD instructions.
    #[inline(always)]
    fn cbrt(&self) -> Self::Output {
        simd_cbrt(self)
    }

    /// Computes floor using platform-optimized SIMD instructions.
    #[inline(always)]
    fn floor(&self) -> Self::Output {
        simd_floor(self)
    }

    /// Computes exponential using platform-optimized SIMD instructions.
    #[inline(always)]
    fn exp(&self) -> Self::Output {
        simd_exp(self)
    }

    /// Computes natural logarithm using platform-optimized SIMD instructions.
    #[inline(always)]
    fn ln(&self) -> Self::Output {
        simd_ln(self)
    }

    /// Computes sine using platform-optimized SIMD instructions.
    #[inline(always)]
    fn sin(&self) -> Self::Output {
        simd_sin(self)
    }

    /// Computes cosine using platform-optimized SIMD instructions.
    /// See `SimdMath<&[f32]> for &[f32]::cos()` for detailed documentation.
    #[inline(always)]
    fn cos(&self) -> Self::Output {
        simd_cos(self)
    }

    /// Computes tangent using platform-optimized SIMD instructions.
    #[inline(always)]
    fn tan(&self) -> Self::Output {
        simd_tan(self)
    }

    /// Computes square root using platform-optimized SIMD instructions.
    #[inline(always)]
    fn sqrt(&self) -> Self::Output {
        simd_sqrt(self)
    }

    /// Computes ceiling using platform-optimized SIMD instructions.
    #[inline(always)]
    fn ceil(&self) -> Self::Output {
        simd_ceil(self)
    }

    /// Computes power function using platform-optimized SIMD instructions.
    #[inline(always)]
    fn pow(&self, other: Self) -> Self::Output {
        simd_pow(self, &other)
    }

    /// Computes 2D hypotenuse using platform-optimized SIMD instructions.
    #[inline(always)]
    fn hypot(&self, other: Self) -> Self::Output {
        simd_hypot(self, &other)
    }

    /// Computes 3D hypotenuse using platform-optimized SIMD instructions.
    #[inline(always)]
    fn hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        simd_hypot3(self, &other1, &other2)
    }

    /// Computes 4D hypotenuse using platform-optimized SIMD instructions.
    #[inline(always)]
    fn hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        simd_hypot4(self, &other1, &other2, &other3)
    }

    // ================================================================================================
    // PARALLEL SIMD METHODS
    // ================================================================================================

    /// Computes absolute value using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_abs(&self) -> Self::Output {
        self.as_slice().par_abs()
    }

    /// Computes arccosine using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_acos(&self) -> Self::Output {
        self.as_slice().par_acos()
    }

    /// Computes arcsine using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_asin(&self) -> Self::Output {
        self.as_slice().par_asin()
    }

    /// Computes arctangent using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_atan(&self) -> Self::Output {
        self.as_slice().par_atan()
    }

    /// Computes two-argument arctangent using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_atan2(&self, other: Self) -> Self::Output {
        self.as_slice().par_atan2(other.as_slice())
    }

    /// Computes cube root using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_cbrt(&self) -> Self::Output {
        self.as_slice().par_cbrt()
    }

    /// Computes ceiling using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_ceil(&self) -> Self::Output {
        self.as_slice().par_ceil()
    }

    /// Computes cosine using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_cos(&self) -> Self::Output {
        self.as_slice().par_cos()
    }

    /// Computes exponential function using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_exp(&self) -> Self::Output {
        self.as_slice().par_exp()
    }

    /// Computes floor using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_floor(&self) -> Self::Output {
        self.as_slice().par_floor()
    }

    /// Computes natural logarithm using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_ln(&self) -> Self::Output {
        self.as_slice().par_ln()
    }

    /// Computes sine using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_sin(&self) -> Self::Output {
        self.as_slice().par_sin()
    }

    /// Computes square root using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_sqrt(&self) -> Self::Output {
        self.as_slice().par_sqrt()
    }

    /// Computes tangent using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_tan(&self) -> Self::Output {
        self.as_slice().par_tan()
    }

    /// Computes 2D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_hypot(&self, other: Self) -> Self::Output {
        self.as_slice().par_hypot(other.as_slice())
    }

    /// Computes 3D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        self.as_slice()
            .par_hypot3(other1.as_slice(), other2.as_slice())
    }

    /// Computes 4D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        self.as_slice()
            .par_hypot4(other1.as_slice(), other2.as_slice(), other3.as_slice())
    }

    /// Computes power function using size-adaptive parallel SIMD.
    ///
    /// Delegates to slice implementation with automatic parallel selection.
    #[inline(always)]
    fn par_pow(&self, other: Self) -> Self::Output {
        self.as_slice().par_pow(other.as_slice())
    }
}

/// Implementation of mathematical operations for f32 slices using platform-optimized SIMD.
///
/// This implementation provides vectorized mathematical functions for f32 slices,
/// leveraging the best available SIMD instructions for improved performance on supported hardware.
///
/// # Performance Characteristics
///
/// - **Vectorization**: Most operations process multiple elements simultaneously using available SIMD width
/// - **Custom approximations**: Uses optimized polynomial approximations for transcendental functions
/// - **Memory efficiency**: Minimizes allocation overhead where possible
/// - **Remainder handling**: Safely processes arrays of any size using partial SIMD operations
///
/// # Precision Trade-offs
///
/// SIMD implementations may have slightly different precision characteristics compared
/// to standard library functions:
/// - **Trigonometric functions**: ~1e-5 to 1e-6 accuracy vs libm
/// - **Exponential/logarithmic**: Similar precision with potential range differences
/// - **Basic operations**: Full precision maintained (abs, sqrt, floor, ceil)
///
/// # Usage
///
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let results = data.as_slice().cos(); // Uses platform-optimized SIMD cosine
/// ```
impl SimdMath for &[f32] {
    type Output = Vec<f32>;

    /// Computes absolute value of each element using platform-optimized SIMD.
    fn abs(&self) -> Self::Output {
        simd_abs(self)
    }

    /// Computes arccosine of each element using platform-optimized SIMD.
    fn acos(&self) -> Self::Output {
        simd_acos(self)
    }

    /// Computes arcsine of each element using platform-optimized SIMD.
    fn asin(&self) -> Self::Output {
        simd_asin(self)
    }

    /// Computes arctangent of each element using platform-optimized SIMD.
    fn atan(&self) -> Self::Output {
        simd_atan(self)
    }

    /// Computes two-argument arctangent using platform-optimized SIMD.
    fn atan2(&self, other: Self) -> Self::Output {
        simd_atan2(self, other)
    }

    /// Computes cube root of each element using platform-optimized SIMD.
    fn cbrt(&self) -> Self::Output {
        simd_cbrt(self)
    }

    /// Computes floor of each element using platform-optimized SIMD.
    fn floor(&self) -> Self::Output {
        simd_floor(self)
    }

    /// Computes exponential of each element using platform-optimized SIMD.
    fn exp(&self) -> Self::Output {
        simd_exp(self)
    }

    /// Computes natural logarithm of each element using platform-optimized SIMD.
    fn ln(&self) -> Self::Output {
        simd_ln(self)
    }

    /// Computes sine of each element using platform-optimized SIMD.
    fn sin(&self) -> Self::Output {
        simd_sin(self)
    }

    /// Computes cosine of each element using platform-optimized SIMD instructions.
    fn cos(&self) -> Self::Output {
        simd_cos(self)
    }

    /// Computes tangent of each element using platform-optimized SIMD.
    fn tan(&self) -> Self::Output {
        simd_tan(self)
    }

    /// Computes square root of each element using platform-optimized SIMD.
    fn sqrt(&self) -> Self::Output {
        simd_sqrt(self)
    }

    /// Computes ceiling of each element using platform-optimized SIMD.
    fn ceil(&self) -> Self::Output {
        simd_ceil(self)
    }

    /// Computes power function using platform-optimized SIMD.
    fn pow(&self, other: Self) -> Self::Output {
        simd_pow(self, other)
    }

    /// Computes 2D hypotenuse using platform-optimized SIMD.
    fn hypot(&self, other: Self) -> Self::Output {
        simd_hypot(self, other)
    }

    /// Computes 3D hypotenuse using platform-optimized SIMD.
    fn hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        simd_hypot3(self, other1, other2)
    }

    /// Computes 4D hypotenuse using platform-optimized SIMD.
    fn hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        simd_hypot4(self, other1, other2, other3)
    }

    // ================================================================================================
    // PARALLEL SIMD METHODS
    // ================================================================================================

    /// Computes absolute value using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size:
    fn par_abs(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_abs(self),
            false => simd_abs(self),
        }
    }

    /// Computes arccosine using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_acos(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_acos(self),
            false => simd_acos(self),
        }
    }

    /// Computes arcsine using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_asin(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_asin(self),
            false => simd_asin(self),
        }
    }

    /// Computes arctangent using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_atan(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_atan(self),
            false => simd_atan(self),
        }
    }

    /// Computes two-argument arctangent using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_atan2(&self, other: Self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_atan2(self, other),
            false => simd_atan2(self, other),
        }
    }

    /// Computes cube root using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_cbrt(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_cbrt(self),
            false => simd_cbrt(self),
        }
    }

    /// Computes ceiling using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_ceil(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_ceil(self),
            false => simd_ceil(self),
        }
    }

    /// Computes cosine using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size:
    fn par_cos(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_cos(self),
            false => simd_cos(self),
        }
    }

    /// Computes exponential function using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_exp(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_exp(self),
            false => simd_exp(self),
        }
    }

    /// Computes floor using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_floor(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_floor(self),
            false => simd_floor(self),
        }
    }

    /// Computes natural logarithm using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_ln(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_ln(self),
            false => simd_ln(self),
        }
    }

    /// Computes sine using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size:
    /// - Arrays ≤ PARALLEL_SIMD_THRESHOLD: Uses regular SIMD for lower latency
    /// - Arrays > PARALLEL_SIMD_THRESHOLD: Uses parallel SIMD for higher throughput
    fn par_sin(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_sin(self),
            false => simd_sin(self),
        }
    }

    /// Computes square root using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_sqrt(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_sqrt(self),
            false => simd_sqrt(self),
        }
    }

    /// Computes tangent using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_tan(&self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_tan(self),
            false => simd_tan(self),
        }
    }

    /// Computes 2D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_hypot(&self, other: Self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_hypot(self, other),
            false => simd_hypot(self, other),
        }
    }

    /// Computes 3D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_hypot3(&self, other1: Self, other2: Self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_hypot3(self, other1, other2),
            false => simd_hypot3(self, other1, other2),
        }
    }

    /// Computes 4D Euclidean distance using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_hypot4(self, other1, other2, other3),
            false => simd_hypot4(self, other1, other2, other3),
        }
    }

    /// Computes power function using size-adaptive parallel SIMD.
    ///
    /// Automatically selects between regular SIMD and parallel SIMD based on array size.
    fn par_pow(&self, other: Self) -> Self::Output {
        match self.len() > PARALLEL_SIMD_THRESHOLD {
            true => parallel_simd_pow(self, other),
            false => simd_pow(self, other),
        }
    }
}
