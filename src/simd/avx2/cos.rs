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

// --- High-precision constants for argument reduction ---
// We represent π/2 as a sum of three f32s to emulate higher precision.
// This is crucial for avoiding catastrophic cancellation with large inputs.
// pi/2 = PI_A + PI_B + PI_C
const PI_A: f32 = 1.5707964f32; // The main f32 part of pi/2
const PI_B: f32 = -4.371139e-8f32; // The first error term
const PI_C: f32 = -2.7118834e-17f32; // The second error term

// --- Minimax Polynomial Coefficients for sin(x)/x on [-π/4, π/4] ---
// These coefficients are optimized to minimize the maximum error for f32.
// We approximate sin(x) ≈ x * (1 + P(x^2)), where P is this polynomial.
const S1: f32 = -1.6666667e-1;
const S2: f32 = 8.333331e-3;
const S3: f32 = -1.9840874e-4;
const S4: f32 = 2.7525562e-6;
const S5: f32 = -2.502943e-8;

/// Calculates the cosine of 8 packed f32 values with high precision using AVX2 and FMA.
///
/// This version improves precision significantly by using:
/// 1. A three-part constant for π/2 to ensure accurate argument reduction.
/// 2. Minimax polynomial coefficients optimized for f32 accuracy.
///
/// The algorithm remains `cos(x) = sin(π/2 - x)`, mapping the problem to a sine
/// approximation on a small interval.
#[target_feature(enable = "avx,avx2,fma")]
pub(crate) unsafe fn _mm256_cos_ps(s: __m256) -> __m256 {
    // --- Handle Special Cases: Infinity and NaN ---
    // An input is invalid if it is NaN or +/- infinity.
    // abs(s) >= infinity covers both cases.
    let abs_s = _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), s);
    let invalid_lanes_mask = _mm256_cmp_ps(abs_s, _mm256_set1_ps(f32::INFINITY), _CMP_GE_OQ);

    // --- Step 1: Range Reduction (High Precision) ---
    // Find j = round(s / π - 0.5)
    let j_float = _mm256_round_ps(
        _mm256_fmadd_ps(
            s,
            _mm256_set1_ps(std::f32::consts::FRAC_1_PI),
            _mm256_set1_ps(-0.5),
        ),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let j = _mm256_cvtps_epi32(j_float);

    // --- Step 2: Calculate the reduced argument r = s - q * π/2 ---
    // q = 2*j + 1
    let q = _mm256_add_epi32(_mm256_slli_epi32(j, 1), _mm256_set1_epi32(1));
    let qf = _mm256_cvtepi32_ps(q);

    // Payne-Hanek style reduction using the high-precision multi-part π/2.
    // This is the key precision improvement for large inputs.
    let mut r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PI_A), s);
    r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PI_B), r);
    r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PI_C), r);

    // --- Step 3: Quadrant Correction ---
    // Determine the sign of the sine argument based on the quadrant.
    // cos((j+0.5)π + r) = sin(r) if j is odd, and -sin(r) if j is even.
    // This is equivalent to sin(r) or sin(-r).
    let j_is_even_mask_i = _mm256_cmpeq_epi32(
        _mm256_and_si256(q, _mm256_set1_epi32(2)),
        _mm256_setzero_si256(),
    );

    let r_neg = _mm256_xor_ps(r, _mm256_set1_ps(-0.0f32)); // Negate r
    let r_signed = _mm256_blendv_ps(r, r_neg, _mm256_castsi256_ps(j_is_even_mask_i));

    // --- Step 4: Polynomial Approximation for sin(r_signed) ---
    // We approximate sin(x) ≈ x * (1 + S1*y + S2*y^2 + S3*y^3 + S4*y^4 + S5*y^5)
    // where y = x^2. This is evaluated with Horner's method.
    let r2 = _mm256_mul_ps(r_signed, r_signed); // y = r_signed^2

    // P(y) = ((((S5*y + S4)*y + S3)*y + S2)*y + S1)
    let mut poly = _mm256_set1_ps(S5);
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(S4));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(S3));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(S2));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(S1));

    // Final result: sin(x) ≈ x + x * x^2 * P(x^2)
    let finite_cos_result = _mm256_fmadd_ps(_mm256_mul_ps(r_signed, r2), poly, r_signed);

    // --- Step 5: Final Blending ---
    // If the original input was invalid (inf/nan), replace the result with NaN.
    let nan_vals = _mm256_set1_ps(f32::NAN);
    _mm256_blendv_ps(finite_cos_result, nan_vals, invalid_lanes_mask)
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

    let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT);

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

    let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT);

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
