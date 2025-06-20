use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub(crate) mod float32 {

    use crate::simd::{
        neon::f32x4::{self, F32x4},
        traits::{SimdAsin, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    use super::*;

    // Coefficients for the asin(x)/x polynomial in x^2 for f32.
    // S(y) = C0 + C1*y + C2*y^2 + C3*y^3 + C4*y^4 + C5*y^5 where y = x*x.
    const ASIN_POLY_1_S: f32 = 1.0f32;

    #[allow(clippy::excessive_precision)]
    const ASIN_POLY_3_S: f32 = 0.1666666716337204f32;
    #[allow(clippy::excessive_precision)]
    const ASIN_POLY_5_S: f32 = 0.07500045746564865f32;
    #[allow(clippy::excessive_precision)]
    const ASIN_POLY_7_S: f32 = 0.04466100037097931f32;
    #[allow(clippy::excessive_precision)]
    const ASIN_POLY_9_S: f32 = 0.03043006919324398f32;
    #[allow(clippy::excessive_precision)]
    const ASIN_POLY_11_S: f32 = 0.022372109815478325f32;

    #[inline]
    /// Computes arcsin for four f32 lanes with high precision using f32 arithmetic.
    /// This implementation aims to minimize floating-point error for f32 inputs
    /// using a polynomial approximation.
    /// Uses vbslq_f32 for copysign functionality.
    pub(crate) unsafe fn vasinq_f32(d: float32x4_t) -> float32x4_t {
        let ones = vdupq_n_f32(1.0f32);
        let abs_d = vabsq_f32(d);

        let nan_mask = vcgtq_f32(abs_d, ones);
        let range_reduction_mask = vcgeq_f32(abs_d, vdupq_n_f32(0.5f32));

        let x_if_reduced = vsqrtq_f32(vdivq_f32(vsubq_f32(ones, abs_d), vdupq_n_f32(2.0f32)));
        let x_for_poly = vbslq_f32(range_reduction_mask, x_if_reduced, abs_d);

        let x_sq = vmulq_f32(x_for_poly, x_for_poly);

        let mut poly_s_val = vdupq_n_f32(ASIN_POLY_11_S);
        poly_s_val = vmlaq_f32(vdupq_n_f32(ASIN_POLY_9_S), poly_s_val, x_sq);
        poly_s_val = vmlaq_f32(vdupq_n_f32(ASIN_POLY_7_S), poly_s_val, x_sq);
        poly_s_val = vmlaq_f32(vdupq_n_f32(ASIN_POLY_5_S), poly_s_val, x_sq);
        poly_s_val = vmlaq_f32(vdupq_n_f32(ASIN_POLY_3_S), poly_s_val, x_sq);
        poly_s_val = vmlaq_f32(vdupq_n_f32(ASIN_POLY_1_S), poly_s_val, x_sq);

        let poly_approx = vmulq_f32(poly_s_val, x_for_poly);

        let val_from_reduced_range = vmlaq_f32(
            vdupq_n_f32(std::f32::consts::FRAC_PI_2),
            vdupq_n_f32(-2.0f32),
            poly_approx,
        );

        let mut result_magnitude =
            vbslq_f32(range_reduction_mask, val_from_reduced_range, poly_approx);

        // Handle NaN for |d| > 1
        result_magnitude = vbslq_f32(nan_mask, vdupq_n_f32(f32::NAN), result_magnitude);

        // Restore original sign using vbslq_f32 for copysign(result_magnitude, d)
        // The sign bit is the MSB (0x80000000 for f32).
        // vbslq_f32(mask, a, b) selects bits from 'a' where mask is 1, and from 'b' where mask is 0.
        // We want the sign bit from 'd' and all other bits from 'result_magnitude'.
        let sign_mask_u32 = vdupq_n_u32(0x80000000u32);

        // result = (d & sign_mask) | (result_magnitude & ~sign_mask)
        // Since result_magnitude is already positive (or NaN), its sign bit is 0.
        // So, (result_magnitude & ~sign_mask) is just result_magnitude.
        // We can use vbslq_f32(sign_mask_f32, d, result_magnitude)
        // This takes the sign bit from 'd' and the remaining bits from 'result_magnitude'.

        vbslq_f32(sign_mask_u32, d, result_magnitude)
    }

    #[inline(always)]
    fn scalar_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.asin()).collect()
    }

    #[target_feature(enable = "neon")]
    fn simd_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        for i in (0..nb_lanes).step_by(step) {
            simd_asin_block(&a[i], &mut c[i]);
        }

        if rem_lanes > 0 {
            simd_asin_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    #[inline(always)]
    fn simd_asin_block(a: *const f32, c: *mut f32) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
        unsafe { a_chunk_simd.asin().store_at(c) };
    }

    #[inline(always)]
    fn simd_asin_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
        unsafe { a_chunk_simd.asin().store_at_partial(c) };
    }

    #[target_feature(enable = "neon")]
    fn parallel_simd_asin(a: &[f32]) -> Vec<f32> {
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
                simd_asin_block(&a[i * step], &mut c_chunk[0]);
            });

        // Handle the remaining elements that do not fit into a full block
        if rem_lanes > 0 {
            simd_asin_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    impl SimdAsin<&[f32]> for &[f32] {
        type Output = Vec<f32>;

        #[inline(always)]
        fn simd_asin(self) -> Self::Output {
            unsafe { simd_asin(self) }
        }

        #[inline(always)]
        fn par_simd_asin(self) -> Self::Output {
            unsafe { parallel_simd_asin(self) }
        }

        #[inline(always)]
        fn scalar_asin(self) -> Self::Output {
            scalar_asin(self)
        }
    }
}
