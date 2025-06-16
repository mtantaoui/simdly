#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) mod float32 {
    use super::*;

    use rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSliceMut,
    };

    use crate::simd::{
        avx512::f32x16::{self, F32x16},
        traits::{SimdAsin, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    // The polynomial coefficients remain the same.
    const P0: f32 = 0.16666666666666666_f32;
    const P1: f32 = 0.07500000000000000_f32;
    const P2: f32 = 0.04464285714285714_f32;
    const P3: f32 = 0.03038181818181818_f32;
    const P4: f32 = 0.02237216981132075_f32;
    const P5: f32 = 0.01735973154362416_f32;
    const P6: f32 = 0.01339831339831340_f32;

    /// Computes `asin(d)` for 16 floats using AVX-512F and FMA intrinsics.
    #[inline(always)]
    pub(crate) unsafe fn _mm512_asin_ps(d: __m512) -> __m512 {
        // AVX-512 uses `__m512` for 16xf32 and `_mm512_` prefixed intrinsics.
        let sign_mask = _mm512_set1_ps(-0.0f32);
        let ones = _mm512_set1_ps(1.0f32);
        let half = _mm512_set1_ps(0.5f32);
        let two = _mm512_set1_ps(2.0f32);
        let pi_2 = _mm512_set1_ps(std::f32::consts::FRAC_PI_2);
        let nan = _mm512_set1_ps(f32::NAN);

        let abs_d = _mm512_andnot_ps(sign_mask, d);

        // Comparison intrinsics now return a `__mmask16` bitmask.
        let nan_mask: __mmask16 = _mm512_cmp_ps_mask(abs_d, ones, _CMP_GT_OS);
        let is_ge_05_mask: __mmask16 = _mm512_cmp_ps_mask(abs_d, half, _CMP_GE_OS);

        // Range reduction logic is identical, using AVX-512 intrinsics.
        let reduced_val = _mm512_sqrt_ps(_mm512_div_ps(_mm512_sub_ps(ones, abs_d), two));

        // Use `_mm512_mask_blend_ps` which takes the mask directly.
        // It selects `reduced_val` where the mask bit is 1, and `abs_d` where it is 0.
        let x = _mm512_mask_blend_ps(is_ge_05_mask, abs_d, reduced_val);

        let x2 = _mm512_mul_ps(x, x);

        // Horner's method using AVX-512 FMA.
        let mut p = _mm512_set1_ps(P6);
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P5));
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P4));
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P3));
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P2));
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P1));
        p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(P0));

        let poly_result = _mm512_fmadd_ps(_mm512_mul_ps(p, x2), x, x);

        // Reconstruction using FNMADD and blending with the mask.
        let reconstructed_res = _mm512_fnmadd_ps(two, poly_result, pi_2);
        let abs_res = _mm512_mask_blend_ps(is_ge_05_mask, poly_result, reconstructed_res);

        // Restore sign and handle NaNs.
        let sign_bits = _mm512_and_ps(d, sign_mask);
        let signed_res = _mm512_or_ps(abs_res, sign_bits);

        _mm512_mask_blend_ps(nan_mask, signed_res, nan)
    }

    #[inline(always)]
    fn scalar_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.asin()).collect()
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn simd_asin(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x16::AVX512_ALIGNMENT);

        let step = f32x16::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x16::load(a, f32x16::LANE_COUNT) };
        unsafe { a_chunk_simd.asin().store_at(c) };
    }

    #[inline(always)]
    fn simd_asin_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x16::load_partial(a, size) };
        unsafe { a_chunk_simd.asin().store_at_partial(c) };
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn parallel_simd_asin(a: &[f32]) -> Vec<f32> {
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
