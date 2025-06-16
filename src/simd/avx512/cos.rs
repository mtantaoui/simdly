#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub(crate) mod float32 {
    use super::*;

    use crate::simd::{
        avx512::f32x16::{self, F32x16},
        traits::{SimdCos, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    // Polynomial coefficients for sine approximation on [-pi/4, pi/4]
    const SIN_POLY_1: f32 = -0.16666666f32;
    const SIN_POLY_2: f32 = 0.008_333_321_f32;
    const SIN_POLY_3: f32 = -0.00019839334f32;
    const SIN_POLY_4: f32 = 2.7522204e-6f32;

    // Polynomial coefficients for cosine approximation on [-pi/4, pi/4]
    const COS_POLY_1: f32 = -0.5f32;
    const COS_POLY_2: f32 = 0.04166662f32;
    const COS_POLY_3: f32 = -0.0013888434f32;
    const COS_POLY_4: f32 = 2.4789852e-5f32;

    // High-precision representation of pi/2 for argument reduction.
    // These constants are derived by successively taking the error of the f32 representation.
    // pi/2 = PI_A + PI_B + PI_C

    // PI_A: The closest f32 value to pi/2.
    const PI_A: f32 = 1.5707964f32; // Equivalent to std::f32::consts::FRAC_PI_2

    // PI_B: The first error term, i.e., f32( (pi/2)_f64 - PI_A_f64 ).
    const PI_B: f32 = -4.371139e-8f32;

    // PI_C: The second error term, i.e., f32( ((pi/2)_f64 - PI_A_f64) - PI_B_f64 ).
    const PI_C: f32 = -2.7118834e-17f32;

    /// Computes the cosine of 16 single-precision floats using AVX-512.
    ///
    /// This implementation uses a standard algorithm:
    /// 1.  **Handle Special Cases:** Inputs like NaN or infinity are masked and result in NaN.
    /// 2.  **Argument Reduction:** The input `d` is reduced to a small range `r` in `[-pi/4, pi/4]`,
    ///     and the quadrant `q` is determined such that `d ≈ q * (pi/2) + r`.
    /// 3.  **Polynomial Approximation:** Depending on the quadrant, we need to compute `cos(r)` or `sin(r)`.
    ///     Two minimax polynomials are used for high accuracy.
    /// 4.  **Quadrant Selection:** The final result is selected based on `q`:
    ///     - `cos(d) ≈ cos(r)` if `q` is even.
    ///     - `cos(d) ≈ -sin(r)` if `q` is odd.
    ///       The sign is also determined by the quadrant.
    ///
    /// This function requires the `avx512f` and `fma` target features.
    #[inline(always)]
    pub(crate) unsafe fn _mm512_cos_ps(d: __m512) -> __m512 {
        // --- Step 1: Handle Special Cases (NaN and Infinity) ---
        let abs_d = _mm512_abs_ps(d);
        let is_invalid_mask = _mm512_cmp_ps_mask(abs_d, _mm512_set1_ps(f32::INFINITY), _CMP_GE_OQ);

        // --- Step 2: Argument Reduction ---
        let q = _mm512_roundscale_ps(
            _mm512_mul_ps(d, _mm512_set1_ps(std::f32::consts::FRAC_2_PI)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let q_int = _mm512_cvtps_epi32(q);

        // Payne-Hanek style reduction using the new, more precise constants.
        // r = d - q * pi/2
        // FMA chain: r = d - q*PI_A - q*PI_B - q*PI_C
        let mut r = _mm512_fmadd_ps(q, _mm512_set1_ps(-PI_A), d);
        r = _mm512_fmadd_ps(q, _mm512_set1_ps(-PI_B), r);
        r = _mm512_fmadd_ps(q, _mm512_set1_ps(-PI_C), r);

        // --- Step 3: Polynomial Approximation ---
        // (No changes here, as it operates on the reduced argument `r`)
        let r2 = _mm512_mul_ps(r, r);

        // Sine polynomial...
        let sin_poly = _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_fmadd_ps(_mm512_set1_ps(SIN_POLY_4), r2, _mm512_set1_ps(SIN_POLY_3)),
                r2,
                _mm512_set1_ps(SIN_POLY_2),
            ),
            r2,
            _mm512_set1_ps(SIN_POLY_1),
        );
        let sin_r = _mm512_fmadd_ps(sin_poly, _mm512_mul_ps(r2, r), r);

        // Cosine polynomial...
        let cos_poly = _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_fmadd_ps(_mm512_set1_ps(COS_POLY_4), r2, _mm512_set1_ps(COS_POLY_3)),
                r2,
                _mm512_set1_ps(COS_POLY_2),
            ),
            r2,
            _mm512_set1_ps(COS_POLY_1),
        );
        let cos_r = _mm512_fmadd_ps(cos_poly, r2, _mm512_set1_ps(1.0));

        // --- Step 4: Quadrant-based Selection ---
        // (No changes here)
        let use_sin_poly_mask = _mm512_test_epi32_mask(q_int, _mm512_set1_epi32(1));
        let mut res = _mm512_mask_blend_ps(use_sin_poly_mask, cos_r, sin_r);

        let q_plus_1 = _mm512_add_epi32(q_int, _mm512_set1_epi32(1));
        let should_negate_mask = _mm512_test_epi32_mask(q_plus_1, _mm512_set1_epi32(2));
        let neg_res = _mm512_xor_ps(res, _mm512_set1_ps(-0.0f32));
        res = _mm512_mask_blend_ps(should_negate_mask, res, neg_res);

        // --- Step 5: Final Blending ---
        // (No changes here)
        let nan_vec = _mm512_set1_ps(f32::NAN);
        _mm512_mask_blend_ps(is_invalid_mask, res, nan_vec)
    }

    #[inline(always)]
    fn scalar_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.cos()).collect()
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x16::AVX512_ALIGNMENT);

        let step = f32x16::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x16::load(a, f32x16::LANE_COUNT) };
        unsafe { a_chunk_simd.cos().store_at(c) };
    }

    #[inline(always)]
    fn simd_cos_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x16::load_partial(a, size) };
        unsafe { a_chunk_simd.cos().store_at_partial(c) };
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn parallel_simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x16::AVX512_ALIGNMENT);

        let step = f32x16::LANE_COUNT;

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
