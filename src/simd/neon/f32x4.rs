#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::ops::Add;

use crate::simd::traits::SimdVec;

pub const NEON_ALIGNMENT: usize = 16;

pub const LANE_COUNT: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    size: usize,
    elements: float32x4_t,
}

impl SimdVec<f32> for F32x4 {
    /// The number of lanes in the vector
    #[inline(always)]
    fn new(slice: &[f32]) -> Self {
        assert!(!slice.is_empty(), "Size can't be empty (size zero)");

        // If the slice length is less than LANE_COUNT, load a partial vector
        // Otherwise, load a full vector
        // This ensures that the vector is always created with the correct number of elements
        // and that the remaining elements are filled with zeros if necessary.
        // This is done to avoid unnecessary branching and ensure all elements are set correctly.
        // The `match` statement is used to handle the different cases based on the slice length.
        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    /// Creates a new vector with all elements set to the same value.
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Self {
            elements: unsafe { vdupq_n_f32(value) },
            size: LANE_COUNT,
        }
    }

    /// Checks if the pointer is aligned.
    #[inline(always)]
    fn is_aligned(_ptr: *const f32) -> bool {
        unreachable!()
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        // Asserts that the pointer is not null and the size is exactly 4 elements.
        // If the pointer is null or the size is not 4, it will panic with an error message.
        assert!(!ptr.is_null(), "Pointer must not be null");
        assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");

        Self {
            elements: unsafe { vld1q_f32(ptr) },
            size,
        }
    }

    /// Loads a vector from an aligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_aligned(_ptr: *const f32, _size: usize) -> Self {
        unreachable!()
    }

    /// Loads a vector from an unaligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_unaligned(_ptr: *const f32, _size: usize) -> Self {
        unreachable!()
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );

        // to avoid unnecessary branching and ensure all elements are set correctly
        assert!(!ptr.is_null(), "Pointer must not be null");

        let elements = match size {
            1 => {
                let v = vdupq_n_f32(0.0);
                vsetq_lane_f32(*ptr.add(0), v, 0)
            }
            2 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                vsetq_lane_f32(*ptr.add(1), v, 1)
            }
            3 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                v = vsetq_lane_f32(*ptr.add(1), v, 1);
                vsetq_lane_f32(*ptr.add(2), v, 2)
            }
            _ => unreachable!("Size must be < {}", LANE_COUNT),
        };

        Self { elements, size }
    }

    #[inline(always)]
    unsafe fn store_in_vec(&self) -> Vec<f32> {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );

        let mut vec = Vec::with_capacity(LANE_COUNT);

        unsafe {
            vst1q_f32(vec.as_mut_ptr(), self.elements);
            vec.set_len(LANE_COUNT);
        }

        vec
    }

    #[inline(always)]
    unsafe fn store_in_vec_partial(&self) -> Vec<f32> {
        match self.size {
            1..LANE_COUNT => unsafe { self.store_in_vec().into_iter().take(self.size).collect() },
            _ => {
                let msg = "Size must be < LANE_COUNT";
                panic!("{}", msg);
            }
        }
    }

    #[inline(always)]
    unsafe fn store_at(&self, ptr: *mut f32) {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

        vst1q_f32(ptr, self.elements);
    }

    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

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
            _ => {
                unreachable!("Size must be < {}", LANE_COUNT)
            }
        }
    }

    #[inline(always)]
    fn to_vec(self) -> Vec<f32> {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );

        if self.size == LANE_COUNT {
            unsafe { self.store_in_vec() }
        } else {
            unsafe { self.store_in_vec_partial() }
        }
    }

    #[inline(always)]
    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a == b elementwise
        let mask = unsafe { vceqq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<b elementwise
        let mask = unsafe { vcltq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn le_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<=b elementwise
        let mask = unsafe { vcleq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>b elementwise
        let mask = unsafe { vcgtq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>=b elementwise
        let mask = unsafe { vcgeq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    unsafe fn cos(&self) -> Self {
        Self {
            elements: vcosq_f32(self.elements),
            size: self.size,
        }
    }
}

/// Implementing the `Add` and `AddAssign` traits for F32x8
/// This allows for using the `+` operator and `+=` operator with F32x8 vectors.    
impl Add for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the add function to perform element-wise addition
        Self {
            size: self.size,
            elements: unsafe { vaddq_f32(self.elements, rhs.elements) },
        }
    }
}

// --- Constants for f32 (Single Precision) ---

// For range reduction, we use a two-part representation of PI.
// This is a common technique (Payne-Hanek style) to maintain precision.
// PI = PI_A + PI_B
#[allow(clippy::approx_constant)]
const PI_A_F32: f32 = 3.1415927; // The high part of PI for f32.
const PI_B_F32: f32 = -8.742278e-8; // The low part (error) of PI for f32.

// Polynomial coefficients for approximating sin(r) for r in [-pi/2, pi/2].
// The polynomial is in terms of r^2 and approximates (sin(r)/r - 1) / r^2.
// P(x) = C1 + C2*x + C3*x^2, where x = r^2
// sin(r) is then reconstructed as: r + r^3 * P(r^2)
#[allow(clippy::excessive_precision)]
const SIN_POLY_1_F: f32 = -0.166666546; // -1/3!
#[allow(clippy::excessive_precision)]
const SIN_POLY_2_F: f32 = 0.00833216087; //  1/5!
const SIN_POLY_3_F: f32 = -0.00019515296; // -1/7!

/// Computes the cosine of four `f32` values in a vector.
///
/// This function implements `cos(d)` by reducing the argument `d` to a value `r`
/// in the range `[-π/2, π/2]` and then using the identity:
///   cos(d) = cos(nπ + r) = (-1)^n * cos(r)
///
/// To use a single high-precision sine polynomial, this is further transformed.
/// The provided f64 implementation uses `cos(d) = sin(π/2 - d)`. This version
/// follows the same logic, reducing `d` such that `d = (n + 1/2)π + r`.
/// This gives `cos(d) = (-1)^(n+1) * sin(r)`.
///
/// # Safety
///
/// This function is safe to call only on AArch64 targets with NEON support.
#[inline(always)]
pub unsafe fn vcosq_f32(d: float32x4_t) -> float32x4_t {
    // --- 1. Range Reduction ---
    // We want to find an integer `n` and a remainder `r` such that:
    // d = (n + 0.5) * π + r
    // `n` is calculated as `round(d/π - 0.5)`.
    let half = vdupq_n_f32(0.5);
    let n = vcvtaq_s32_f32(vsubq_f32(vmulq_n_f32(d, std::f32::consts::FRAC_1_PI), half));

    // Now we compute r = d - (n + 0.5) * π.
    // To maintain precision, we use the two-part PI representation.
    // r = d - (n+0.5)*PI_A - (n+0.5)*PI_B
    let n_plus_half = vaddq_f32(vcvtq_f32_s32(n), half);

    // r = d - (n+0.5) * PI_A
    let mut r = vmlsq_f32(d, n_plus_half, vdupq_n_f32(PI_A_F32));
    // r = r - (n+0.5) * PI_B
    r = vmlsq_f32(r, n_plus_half, vdupq_n_f32(PI_B_F32));

    // --- 2. Sign Correction ---
    // The result is `cos(d) = (-1)^(n+1) * sin(r)`.
    // The sign depends on `n+1`. The polynomial computes `sin(r)`.
    // We can fold the sign into `r` before the polynomial evaluation.
    // If `n+1` is odd, sign is negative. `(n+1) & 1 != 0`.
    // This is equivalent to `n` being even.

    // Create a sign mask where bits are set if n is even.
    let n_is_even_mask = vceqq_s32(vandq_s32(n, vdupq_n_s32(1)), vdupq_n_s32(0));
    // The sign bit for a float is the most significant bit.
    let sign_bit = vdupq_n_u32(0x80000000);
    let sign_mask = vandq_u32(n_is_even_mask, sign_bit);

    // Flip the sign of `r` if `n` is even. This computes `±r`.
    r = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(r), sign_mask));

    // --- 3. Polynomial Evaluation ---
    // We approximate `sin(r)` using a minimax polynomial for `f32`.
    // The polynomial approximates `(sin(r)/r - 1) / r^2`.
    // P(r^2) = C1 + C2*r^2 + C3*r^4
    // sin(r) ≈ r + r * r^2 * P(r^2) = r + r^3 * P(r^2)

    let x2 = vmulq_f32(r, r); // r^2

    // Evaluate the polynomial P(r^2) using Horner's method.
    // p = C3
    let mut p = vdupq_n_f32(SIN_POLY_3_F);
    // p = C2 + p * x2
    p = vmlaq_f32(vdupq_n_f32(SIN_POLY_2_F), p, x2);
    // p = C1 + p * x2
    p = vmlaq_f32(vdupq_n_f32(SIN_POLY_1_F), p, x2);

    // --- 4. Final Reconstruction ---
    // res = r + r^3 * p = r + (r * r^2) * p
    let r_cubed = vmulq_f32(r, x2);
    let res = vmlaq_f32(r, p, r_cubed);

    res
}
