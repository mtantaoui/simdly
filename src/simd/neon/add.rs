use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    mem,
};

use rayon::prelude::*;

use crate::simd::{
    neon::f32x4::{self, F32x4},
    traits::{SimdAdd, SimdCos, SimdVec},
};

fn alloc_uninit_f32_vec(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout = Layout::from_size_align(len * mem::size_of::<f32>(), f32x4::NEON_ALIGNMENT)
        .expect("Invalid layout");

    let ptr = unsafe { alloc(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

#[inline(always)]
fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline(always)]
pub fn parallel_scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect()
}

#[target_feature(enable = "neon")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    let step = f32x4::LANE_COUNT;

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
            rem_lanes, // number of reminaing uncomplete lanes
        );
    }

    c
}

#[inline(always)]
fn simd_add_block(a: *const f32, b: *const f32, c: *mut f32) {
    // Assumes lengths are f32x4::LANE_COUNT
    let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x4::load(b, f32x4::LANE_COUNT) };
    unsafe { (a_chunk_simd + b_chunk_simd).store_at(c) };
}

#[inline(always)]
fn simd_add_partial_block(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
    // Assumes lengths are f32x4::LANE_COUNT
    let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x4::load_partial(b, size) };
    unsafe { (a_chunk_simd + b_chunk_simd).store_at_partial(c) };
}

#[target_feature(enable = "neon")]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    let step = f32x4::LANE_COUNT;

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

#[inline(always)]
fn scalar_cos(a: &[f32]) -> Vec<f32> {
    assert!(!a.is_empty(), "Size can't be empty (size zero)");

    a.iter().map(|x| x.cos()).collect()
}

#[target_feature(enable = "neon")]
fn simd_cos(a: &[f32]) -> Vec<f32> {
    assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

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

    let mut c = alloc_uninit_f32_vec(size);

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

impl<'b> SimdCos<&'b [f32]> for &[f32] {
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
