use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::simd::{
    neon::f32x4::{self, F32x4},
    traits::{SimdCos, SimdVec},
    utils::alloc_uninit_f32_vec,
};

pub(crate) mod f32 {
    use super::*;

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

    // #[allow(clippy::excessive_precision)]
    // const SIN_POLY_1_F: f32 = -0.16666666666666666; // -1/3!
    // #[allow(clippy::excessive_precision)]
    // const SIN_POLY_2_F: f32 = 8.333333333333333e-3; // +1/5!
    // #[allow(clippy::excessive_precision)]
    // const SIN_POLY_3_F: f32 = -1.984126984126984e-4; // -1/7!

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

        vmlaq_f32(r, p, r_cubed)
    }

    #[inline(always)]
    fn scalar_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.cos()).collect()
    }

    #[target_feature(enable = "neon")]
    fn simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        for i in (0..nb_lanes).step_by(step) {
            simd_cos_block(&a[i], &mut c[i]);
        }

        if rem_lanes > 0 {
            simd_cos_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    #[inline(always)]
    fn simd_cos_block(a: *const f32, c: *mut f32) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
        unsafe { a_chunk_simd.cos().store_at(c) };
    }

    #[inline(always)]
    fn simd_cos_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
        unsafe { a_chunk_simd.cos().store_at_partial(c) };
    }

    #[target_feature(enable = "neon")]
    fn parallel_simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        // Use chunks_exact_mut to ensure we process full blocks of size `step`
        // and handle the remaining elements separately.
        c.par_chunks_exact_mut(step)
            .enumerate()
            .for_each(|(i, c_chunk)| {
                simd_cos_block(&a[i * step], &mut c_chunk[0]);
            });

        // Handle the remaining elements that do not fit into a full block
        if rem_lanes > 0 {
            simd_cos_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    impl SimdCos<&[f32]> for &[f32] {
        type Output = Vec<f32>;

        #[inline(always)]
        fn simd_cos(self) -> Self::Output {
            unsafe { simd_cos(self) }
        }

        #[inline(always)]
        fn par_simd_cos(self) -> Self::Output {
            unsafe { parallel_simd_cos(self) }
        }

        #[inline(always)]
        fn scalar_cos(self) -> Self::Output {
            scalar_cos(self)
        }
    }
}
