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
    // --- Handle Infinity ---
    // Create a mask for lanes that are +INF or -INF
    // 1. Get absolute value of s
    let abs_s = _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), s); // abs(s) by clearing sign bit
                                                              // 2. Compare with +INF
    let inf_lanes_mask = _mm256_cmp_ps(abs_s, _mm256_set1_ps(f32::INFINITY), _CMP_EQ_OQ);
    // inf_lanes_mask now has 0xFFFFFFFF for lanes that were INF, and 0x00000000 otherwise.
    // --- Handle Infinity ---

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
    // For INF input to round_ps, result is INF. cvtps_epi32(INF) yields 0x80000000 (INT_MIN for signed 32-bit).
    // For NaN input to round_ps, result is NaN. cvtps_epi32(NaN) yields 0x80000000.
    // This means the "finite_cos_result" will be garbage for INF/NaN inputs, but we'll mask it out.
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
    // `(q & 2) == 0` is a clever way to check if `j` is even.
    // (q = 2j+1. If j is even, j=2k -> q=4k+1. q&2 = (4k+1)&2 = 0.
    //  If j is odd,  j=2k+1 -> q=4k+3. q&2 = (4k+3)&2 = 2.)
    let j_is_even_mask_i = _mm256_cmpeq_epi32(
        _mm256_and_si256(q, _mm256_set1_epi32(2)),
        _mm256_setzero_si256(),
    );
    // The mask j_is_even_mask_i is an integer mask (0xFFFFFFFF or 0x0).
    // _mm256_castsi256_ps converts it bitwise to a float mask.

    // The sign of the result depends on `j`.
    // cos( (j+0.5)π + r' ) = (-1)^(j+1) * sin(r')
    // If j is even (j_is_even_mask is true), we need (-1)^(even+1) * sin(r') = -sin(r') = sin(-r').
    // If j is odd  (j_is_even_mask is false), we need (-1)^(odd+1) * sin(r') = +sin(r').
    let r_neg = _mm256_xor_ps(r, _mm256_set1_ps(-0.0f32)); // Negate r by flipping sign bit
    let r_signed = _mm256_blendv_ps(r, r_neg, _mm256_castsi256_ps(j_is_even_mask_i));

    // Step 5: Polynomial Approximation for sin(r_signed).
    // sin(x) ≈ x * (1 + C2*x^2 + C3*x^4 + C4*x^6 + C5*x^8 + C6*x^10 + C7*x^12)
    // Polynomial P(y) = C2 + C3*y + C4*y^2 + C5*y^3 + C6*y^4 + C7*y^5 where y = r_signed^2
    // Evaluated using Horner's method:
    // P(y) = ((((C7*y + C6)*y + C5)*y + C4)*y + C3)*y + C2
    let r2 = _mm256_mul_ps(r_signed, r_signed); // r_signed^2

    let mut poly = _mm256_set1_ps(SIN_POLY_7_S);
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_6_S));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_5_S));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_4_S));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_3_S));
    poly = _mm256_fmadd_ps(poly, r2, _mm256_set1_ps(SIN_POLY_2_S));

    // Final result: sin(x) ≈ x + x * x^2 * P(x^2)  (using fmadd: x*x^2 * P(x^2) + x)
    // which is r_signed + r_signed * r2 * poly
    let finite_cos_result = _mm256_fmadd_ps(_mm256_mul_ps(r_signed, r2), poly, r_signed);

    // --- Blend with NaN for Infinity inputs ---
    let nan_vals = _mm256_set1_ps(f32::NAN);

    // --- Blend with NaN for Infinity inputs ---
    _mm256_blendv_ps(finite_cos_result, nan_vals, inf_lanes_mask)
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
