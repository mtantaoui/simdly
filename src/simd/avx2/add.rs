use std::alloc::{alloc, handle_alloc_error, Layout};

use rayon::prelude::*; // For parallel processing;

use crate::simd::avx2::f32x8::{self, F32x8};
use crate::simd::traits::{SimdAdd, SimdVec};

/// Allocates a 32-byte aligned `Vec<f32>` with uninitialized contents.
///
/// # Safety
///
/// The caller must ensure that the elements of the returned vector are
/// initialized before being read. Reading from uninitialized memory is
/// undefined behavior.
#[inline(always)]
fn alloc_uninit_f32_vec(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), f32x8::AVX_ALIGNMENT)
        .expect("Invalid layout");

    let ptr = unsafe { alloc(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: The pointer is non-null and the layout is valid for `len` elements.
    // The capacity is set to `len`, so no re-allocation will occur until it's grown.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

pub fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline(always)]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    for (idx, c_chunk) in c.chunks_mut(8).enumerate() {
        // for idx in (0..size).step_by(8) {

        let i = idx * f32x8::LANE_COUNT;

        let a_chunk = F32x8::new(&a[i..]);
        let b_chunk = F32x8::new(&b[i..]);
        let sum = a_chunk + b_chunk;

        unsafe {
            sum.store_at(c_chunk.as_mut_ptr());
        }
    }

    c
}

#[inline(always)]
fn simd_add_block(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    for (idx, c_chunk) in c.chunks_mut(f32x8::LANE_COUNT).enumerate() {
        let i = idx * f32x8::LANE_COUNT;
        let a_chunk = F32x8::new(&a[i..]);
        let b_chunk = F32x8::new(&b[i..]);
        let sum = a_chunk + b_chunk;

        unsafe {
            sum.store_at(c_chunk.as_mut_ptr());
        }
    }
}

#[inline(always)]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    c.par_chunks_mut(f32x8::AVX_ALIGNMENT)
        .zip(a.par_chunks(f32x8::AVX_ALIGNMENT))
        .zip(b.par_chunks(f32x8::AVX_ALIGNMENT))
        .for_each(|((c_chunk, a_chunk), b_chunk)| {
            simd_add_block(a_chunk, b_chunk, c_chunk);
        });

    c
}

impl<'b> SimdAdd<&'b [f32]> for &[f32] {
    type Output = Vec<f32>;

    #[inline(always)]
    fn simd_add(self, rhs: &'b [f32]) -> Self::Output {
        simd_add(self, rhs)
    }

    #[inline(always)]
    fn par_simd_add(self, rhs: &'b [f32]) -> Self::Output {
        parallel_simd_add(self, rhs)
    }

    #[inline(always)]
    fn scalar_add(self, rhs: &'b [f32]) -> Self::Output {
        scalar_add(self, rhs)
    }
}
