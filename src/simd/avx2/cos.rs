use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::simd::{
    avx2::f32x8::{self, F32x8},
    traits::{SimdCos, SimdVec},
    utils::alloc_uninit_f32_vec,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) mod f32 {
    use super::*;

    // --- Polynomial Coefficients for sin(x) approximation on [-π/4, π/4] ---
    // These are f32 casts of the original f64 coefficients.
    // NOTE: For production-grade accuracy, these should be re-derived specifically
    // for f32 to be optimal (minimax coefficients). These are just for a direct translation.
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_2_S: f32 = -0.16666666666666666; // -1/3!
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_3_S: f32 = 8.333333333333333e-3; // +1/5!
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_4_S: f32 = -1.984126984126984e-4; // -1/7!
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_5_S: f32 = 2.7557319223985893e-6; // +1/9!
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_6_S: f32 = -2.5052108385441718e-8; // -1/11!
    #[allow(clippy::excessive_precision)]
    const SIN_POLY_7_S: f32 = 1.6059043836821613e-10; // -1/13!
                                                      // Fewer coefficients are needed for f32 precision. We can stop here.
                                                      // The original used 10 coefficients for f64's higher precision needs.

    /// Calculates the cosine of 8 packed f32 values using AVX2 and FMA.
    ///
    /// This is an f32 version of the provided f64 function.
    /// It uses a range reduction to [-π/4, π/4] and a polynomial approximation for sin(r).
    #[target_feature(enable = "avx,avx2,fma")]
    pub(crate) unsafe fn _mm256_cos_ps(s: __m256) -> __m256 {
        // Step 1: Range Reduction. Find j = round(s / π - 0.5)
        // This calculates j such that s ≈ (j + 0.5) * π.
        let j_float = _mm256_round_ps(
            _mm256_fmadd_ps(
                s,
                _mm256_set1_ps(std::f32::consts::FRAC_1_PI),
                _mm256_set1_ps(-0.5),
            ),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let j = _mm256_cvtps_epi32(j_float);

        // Step 2: Calculate q = 2*j + 1. This helps map to the correct sine/cosine quadrant.
        // q will be an odd integer.
        let q = _mm256_add_epi32(_mm256_slli_epi32(j, 1), _mm256_set1_epi32(1));
        let qf = _mm256_cvtepi32_ps(q);

        // Step 3: Calculate the reduced argument r = s - q * π/2.
        // We use a fused-multiply-add for precision: s + qf * (-π/2)
        // The result `r` will be in the range [-π/4, π/4].
        let r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-std::f32::consts::FRAC_PI_2), s);

        // Step 4: Quadrant Correction.
        // cos(x) = cos(q*π/2 + r) = sin(π/2 - (q*π/2+r)) = sin((1-q)*π/2 - r)
        // The identity is cos(j*π + α) = (-1)^j * cos(α).
        // Our reduction is different: cos( (j+0.5)π + r' ) = (-1)^(j+1) * sin(r')
        // We check if j is even or odd to determine the sign.
        // `(q & 2) == 0` is a clever way to check if `j` is even.
        let j_is_even_mask = _mm256_cmpeq_epi32(
            _mm256_and_si256(q, _mm256_set1_epi32(2)),
            _mm256_setzero_si256(),
        );

        // The sign of the result depends on `j`.
        // If j is even, cos(...) = -sin(r). If j is odd, cos(...) = +sin(r).
        // We can achieve this by computing sin(r) and negating it if j is even.
        // Or, more efficiently, by computing sin(-r) if j is even and sin(r) if j is odd,
        // since sin(-r) = -sin(r).
        // We use a blend to select `r` or `-r` based on the mask.
        let r_neg = _mm256_xor_ps(r, _mm256_set1_ps(-0.0f32)); // Negate r by flipping sign bit
        let r_signed = _mm256_blendv_ps(r, r_neg, _mm256_castsi256_ps(j_is_even_mask));

        // Step 5: Polynomial Approximation for sin(r_signed).
        // The polynomial approximates sin(x) ≈ x * (1 + c2*x^2 + c3*x^4 + ...)
        // We use Horner's method for efficient evaluation.
        let r2 = _mm256_mul_ps(r_signed, r_signed); // r_signed^2

        // Evaluate the polynomial P(r2) = c7*r2^5 + c6*r2^4 + ... + c2
        let mut poly = _mm256_set1_ps(SIN_POLY_7_S);
        poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_6_S));
        poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_5_S));
        poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_4_S));
        poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_3_S));
        poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_2_S));

        // Final step of Horner's method: sin(x) ≈ x + x * r2 * P(r2)
        // which is equivalent to x * (1 + r2 * P(r2))
        let r2_poly = _mm256_mul_ps(poly, r2);

        _mm256_fmadd_ps(r2_poly, r_signed, r_signed)
    }

    #[inline(always)]
    fn scalar_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.cos()).collect()
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size);

        let step = f32x8::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
        unsafe { a_chunk_simd.cos().store_at(c) };
    }

    #[inline(always)]
    fn simd_cos_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
        unsafe { a_chunk_simd.cos().store_at_partial(c) };
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn parallel_simd_cos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size);

        let step = f32x8::LANE_COUNT;

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
