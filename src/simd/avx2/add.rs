use std::alloc::{alloc, handle_alloc_error, Layout};
use std::arch::x86_64::*;
use std::ptr;

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

#[inline(always)]
pub fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline(always)]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size);

    c.chunks_mut(f32x8::AVX_ALIGNMENT)
        .zip(a.chunks(f32x8::AVX_ALIGNMENT))
        .zip(b.chunks(f32x8::AVX_ALIGNMENT))
        .for_each(|((c_chunk, a_chunk), b_chunk)| {
            simd_add_block(a_chunk, b_chunk, c_chunk);
        });

    c
}

#[inline(always)]
fn simd_add_block(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    c.chunks_mut(f32x8::LANE_COUNT)
        .zip(a.chunks(f32x8::LANE_COUNT))
        .zip(b.chunks(f32x8::LANE_COUNT))
        .for_each(|((c_chunk, a_chunk), b_chunk)| {
            let a_vec = F32x8::new(a_chunk);
            let b_vec = F32x8::new(b_chunk);
            let sum = a_vec + b_vec;
            unsafe {
                sum.store_at(c_chunk.as_mut_ptr());
            }
        });
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

pub fn simd_add_optimized(a: &[f32], b: &[f32]) -> std::vec::Vec<f32> {
    assert_eq!(a.len(), b.len());

    let size = a.len();

    let mut res = alloc_uninit_f32_vec(size);

    let len = a.len();
    let floats_per_vec = 8; // 8 floats in a __m256

    // Use raw pointers for cleaner pointer arithmetic
    let mut a_ptr = a.as_ptr();
    let mut b_ptr = b.as_ptr();
    let mut res_ptr = res.as_mut_ptr();

    // Process the bulk of the data in unrolled chunks of 4 vectors (32 floats)
    let unroll_factor = 4;
    let chunk_size = floats_per_vec * unroll_factor; // 32 floats

    // Number of full chunks we can process
    let num_chunks = len / chunk_size;

    // Use `unsafe` because we are manually managing pointers and using CPU intrinsics.
    // The assertions at the top provide the safety guarantee.
    unsafe {
        for _ in 0..num_chunks {
            // Load 4 vectors from 'a'
            let a1 = _mm256_loadu_ps(a_ptr);
            let a2 = _mm256_loadu_ps(a_ptr.add(floats_per_vec * 1));
            let a3 = _mm256_loadu_ps(a_ptr.add(floats_per_vec * 2));
            let a4 = _mm256_loadu_ps(a_ptr.add(floats_per_vec * 3));

            // Load 4 vectors from 'b'
            let b1 = _mm256_loadu_ps(b_ptr);
            let b2 = _mm256_loadu_ps(b_ptr.add(floats_per_vec * 1));
            let b3 = _mm256_loadu_ps(b_ptr.add(floats_per_vec * 2));
            let b4 = _mm256_loadu_ps(b_ptr.add(floats_per_vec * 3));

            // Perform the additions
            let r1 = _mm256_add_ps(a1, b1);
            let r2 = _mm256_add_ps(a2, b2);
            let r3 = _mm256_add_ps(a3, b3);
            let r4 = _mm256_add_ps(a4, b4);

            // --- CRITICAL OPTIMIZATION: Non-Temporal (Streaming) Stores ---
            // This tells the CPU to write directly to memory, bypassing the cache.
            // This is a huge win for large vector operations.
            _mm256_stream_ps(res_ptr, r1);
            _mm256_stream_ps(res_ptr.add(floats_per_vec * 1), r2);
            _mm256_stream_ps(res_ptr.add(floats_per_vec * 2), r3);
            _mm256_stream_ps(res_ptr.add(floats_per_vec * 3), r4);

            // Advance pointers
            a_ptr = a_ptr.add(chunk_size);
            b_ptr = b_ptr.add(chunk_size);
            res_ptr = res_ptr.add(chunk_size);
        }

        // Process any remaining vectors (less than 4)
        let remaining_start = num_chunks * chunk_size;
        let mut i = remaining_start;

        while i + floats_per_vec <= len {
            let a_vec = _mm256_loadu_ps(a_ptr);
            let b_vec = _mm256_loadu_ps(b_ptr);
            let res_vec = _mm256_add_ps(a_vec, b_vec);
            // Use a standard unaligned store for the smaller remaining part
            _mm256_storeu_ps(res_ptr, res_vec);

            a_ptr = a_ptr.add(floats_per_vec);
            b_ptr = b_ptr.add(floats_per_vec);
            res_ptr = res_ptr.add(floats_per_vec);
            i += floats_per_vec;
        }

        // Handle the final remainder (0-7 floats) with a scalar loop
        // The compiler is very good at optimizing this small final loop.
        while i < len {
            ptr::write(res_ptr, ptr::read(a_ptr) + ptr::read(b_ptr));

            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            res_ptr = res_ptr.add(1);
            i += 1;
        }
    }

    res
}
