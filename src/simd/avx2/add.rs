use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::simd::avx2::f32x8::{self, F32x8};
use crate::simd::traits::{SimdAdd, SimdVec};
use crate::simd::utils::alloc_uninit_f32_vec;

#[inline(always)]
fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[target_feature(enable = "avx,avx2,fma")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    let step = f32x8::LANE_COUNT;

    let nb_lanes = size - (size % step);
    let rem_lanes = size - nb_lanes;

    for i in (0..nb_lanes).step_by(step) {
        simd_add_block(&a[i], &b[i], &mut c[i]);
    }

    if rem_lanes > 0 {
        simd_add_partial_block(
            &a[nb_lanes],
            &b[nb_lanes],
            &mut c[nb_lanes],
            rem_lanes, // number of remainaing uncomplete lanes
        );
    }

    c
}

#[inline(always)]
fn simd_add_block(a: *const f32, b: *const f32, c: *mut f32) {
    // Assumes a and b are aligned to f32x8::AVX_ALIGNMENT
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x8::load(b, f32x8::LANE_COUNT) };
    unsafe { (a_chunk_simd + b_chunk_simd).store_at(c) };
}

#[inline(always)]
fn simd_add_partial_block(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
    // Assumes size is less than f32x8::LANE_COUNT
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x8::load_partial(b, size) };
    unsafe { (a_chunk_simd + b_chunk_simd).store_at_partial(c) };
}

#[target_feature(enable = "avx,avx2,fma")]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

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
            simd_add_block(&a[i * step], &b[i * step], &mut c_chunk[0]);
        });

    // Handle the remaining elements that do not fit into a full block
    if rem_lanes > 0 {
        simd_add_partial_block(
            &a[nb_lanes],
            &b[nb_lanes],
            &mut c[nb_lanes],
            rem_lanes, // number of reminaing uncomplete lanes
        );
    }

    c
}

impl<'b> SimdAdd<&'b [f32]> for &[f32] {
    type Output = Vec<f32>;

    #[inline(always)]
    fn simd_add(self, rhs: &'b [f32]) -> Self::Output {
        unsafe { simd_add(self, rhs) }
    }

    #[inline(always)]
    fn par_simd_add(self, rhs: &'b [f32]) -> Self::Output {
        unsafe { parallel_simd_add(self, rhs) }
    }

    #[inline(always)]
    fn scalar_add(self, rhs: &'b [f32]) -> Self::Output {
        scalar_add(self, rhs)
    }
}
