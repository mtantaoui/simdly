use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    mem,
};

use rayon::prelude::*;

use crate::simd::{
    neon::f32x4::{self, F32x4},
    traits::{SimdAdd, SimdVec},
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
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline(always)]
pub fn parallel_scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect()
}

#[target_feature(enable = "neon")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    let step = f32x4::LANE_COUNT;

    // let mut i = 0;
    // while i < size {
    for i in (0..size).step_by(step) {
        let a_addr = &a[i];
        let b_addr = &b[i];
        let c_addr = &mut c[i];

        simd_add_block(a_addr, b_addr, c_addr);
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

#[target_feature(enable = "neon")]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    let step = f32x4::LANE_COUNT;

    // let mut i = 0;
    // while i < size {

    c.par_chunks_mut(step).enumerate().for_each(|(i, c)| {
        let a_addr = &a[i];
        let b_addr = &b[i];
        let c_addr = &mut c[0];

        simd_add_block(a_addr, b_addr, c_addr);
    });
    // (0..size).into_par_iter().step_by(step).for_each();
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
