#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) mod float32 {
    use rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSliceMut,
    };

    use crate::simd::{
        avx2::f32x8::{self, F32x8},
        traits::{SimdAsin, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    use super::*;

    // Standard coefficients for asin(x)/x approximation using a polynomial in x^2.
    // P(z) = c0 + c1*z + c2*z^2 + c3*z^3 + c4*z^4 + c5*z^5 + c6*z^6, where z = x^2
    #[allow(clippy::excessive_precision)]
    const P0: f32 = 0.16666666666666666_f32;
    #[allow(clippy::excessive_precision)]
    const P1: f32 = 0.07500000000000000_f32;
    #[allow(clippy::excessive_precision)]
    const P2: f32 = 0.04464285714285714_f32;
    #[allow(clippy::excessive_precision)]
    const P3: f32 = 0.03038181818181818_f32;
    #[allow(clippy::excessive_precision)]
    const P4: f32 = 0.02237216981132075_f32;
    #[allow(clippy::excessive_precision)]
    const P5: f32 = 0.01735973154362416_f32;
    #[allow(clippy::excessive_precision)]
    const P6: f32 = 0.01339831339831340_f32;

    #[inline]
    pub(crate) unsafe fn _mm256_asin_ps(d: __m256) -> __m256 {
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let ones = _mm256_set1_ps(1.0f32);
        let half = _mm256_set1_ps(0.5f32);
        let two = _mm256_set1_ps(2.0f32);
        let pi_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        let nan = _mm256_set1_ps(f32::NAN);

        let abs_d = _mm256_andnot_ps(sign_mask, d);
        let nan_mask = _mm256_cmp_ps(abs_d, ones, _CMP_GT_OS);
        let is_ge_05_mask = _mm256_cmp_ps(abs_d, half, _CMP_GE_OS);

        let reduced_val = _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(ones, abs_d), two));
        let x = _mm256_blendv_ps(abs_d, reduced_val, is_ge_05_mask);

        // --- Polynomial in x^2 for asin(x)/x ---
        // Let y = asin(x)/x - 1. We approximate y with x^2 * P(x^2).
        let x2 = _mm256_mul_ps(x, x);

        // Evaluate polynomial P(x^2) using Horner's method
        let mut p = _mm256_set1_ps(P6);
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P5));
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P4));
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P3));
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P2));
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P1));
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(P0));

        // poly_result = x + x * x^2 * P(x^2)
        let poly_result = _mm256_fmadd_ps(_mm256_mul_ps(p, x2), x, x);

        let reconstructed_res = _mm256_fnmadd_ps(two, poly_result, pi_2);
        let abs_res = _mm256_blendv_ps(poly_result, reconstructed_res, is_ge_05_mask);

        let sign_bits = _mm256_and_ps(d, sign_mask);
        let signed_res = _mm256_or_ps(abs_res, sign_bits);

        _mm256_blendv_ps(signed_res, nan, nan_mask)
    }

    #[inline(always)]
    fn scalar_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.asin()).collect()
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn simd_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT);

        let step = f32x8::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
        unsafe { a_chunk_simd.asin().store_at(c) };
    }

    #[inline(always)]
    fn simd_asin_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
        unsafe { a_chunk_simd.asin().store_at_partial(c) };
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn parallel_simd_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT);

        let step = f32x8::LANE_COUNT;

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
