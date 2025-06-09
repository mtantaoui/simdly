use rayon::prelude::*;
use std::alloc::{alloc, handle_alloc_error, Layout};
use std::arch::x86_64::*; // For parallel processing;

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

// #[inline(always)]
pub fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// #[inline(always)]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();
    let mut c = alloc_uninit_f32_vec(size);

    let mut i = 0;
    let step = f32x8::LANE_COUNT;

    while i < size {
        let a_addr = &a[i];
        let b_addr = &b[i];

        let a_chunk = unsafe { F32x8::load(a_addr, step) };
        let b_chunk = unsafe { F32x8::load(b_addr, step) };

        let c_addr = &mut c[i];
        unsafe { (a_chunk + b_chunk).store_at(c_addr) };

        i += step;
    }

    c
}

/// Processes a block of f32 data using AVX intrinsics for addition.
///
/// This function is designed for high performance on a given chunk of data.
/// It uses loop unrolling and handles remainders.
///
/// # Safety
///
/// - The caller must ensure that AVX is supported on the target CPU,
///   or this function must be compiled with `target-feature=+avx`.
/// - Pointers `a_ptr`, `b_ptr`, `c_ptr` derived from slices must be valid for `len` elements.
/// - `c` slice is assumed to contain uninitialized memory and will be fully written.
/// - For optimal performance with streaming stores, the memory pointed to by `c_ptr`
///   should be 32-byte aligned.
#[target_feature(enable = "avx")]
pub unsafe fn simd_add_block(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "Input slices 'a' and 'b' must have the same length."
    );
    assert_eq!(
        a.len(),
        c.len(),
        "Input and output slices must have the same length."
    );

    let len = a.len();
    if len == 0 {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // Main unrolled loop
    // for _ in 0..num_unrolled_chunks {
    // Load 4 vectors from 'a' (unaligned)
    let a1 = _mm256_loadu_ps(a_ptr);
    let a2 = _mm256_loadu_ps(a_ptr.add(f32x8::LANE_COUNT * 1));
    let a3 = _mm256_loadu_ps(a_ptr.add(f32x8::LANE_COUNT * 2));
    let a4 = _mm256_loadu_ps(a_ptr.add(f32x8::LANE_COUNT * 3));

    // Load 4 vectors from 'b' (unaligned)
    let b1 = _mm256_loadu_ps(b_ptr);
    let b2 = _mm256_loadu_ps(b_ptr.add(f32x8::LANE_COUNT * 1));
    let b3 = _mm256_loadu_ps(b_ptr.add(f32x8::LANE_COUNT * 2));
    let b4 = _mm256_loadu_ps(b_ptr.add(f32x8::LANE_COUNT * 3));

    // Perform additions
    let r1 = _mm256_add_ps(a1, b1);
    let r2 = _mm256_add_ps(a2, b2);
    let r3 = _mm256_add_ps(a3, b3);
    let r4 = _mm256_add_ps(a4, b4);

    // // Store results using streaming stores (assuming 'c_ptr' is aligned)
    // // If 'c_ptr' might not be aligned, use _mm256_storeu_ps instead,
    // // but performance will be slightly less for large sequential writes.
    _mm256_store_ps(c_ptr, r1);
    _mm256_store_ps(c_ptr.add(f32x8::LANE_COUNT * 1), r2);
    _mm256_store_ps(c_ptr.add(f32x8::LANE_COUNT * 2), r3);
    _mm256_store_ps(c_ptr.add(f32x8::LANE_COUNT * 3), r4);
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
            unsafe { simd_add_block(a_chunk, b_chunk, c_chunk) };
        });

    c
}

impl<'b> SimdAdd<&'b [f32]> for &[f32] {
    type Output = Vec<f32>;

    // #[inline(always)]
    fn simd_add(self, rhs: &'b [f32]) -> Self::Output {
        simd_add(self, rhs)
    }

    // #[inline(always)]
    fn par_simd_add(self, rhs: &'b [f32]) -> Self::Output {
        parallel_simd_add(self, rhs)
    }

    // #[inline(always)]
    fn scalar_add(self, rhs: &'b [f32]) -> Self::Output {
        scalar_add(self, rhs)
    }
}

#[target_feature(enable = "avx")]
pub unsafe fn simd_add_optimized_store(a: &[f32], b: &[f32]) -> std::vec::Vec<f32> {
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
            _mm256_store_ps(res_ptr, r1);
            _mm256_store_ps(res_ptr.add(floats_per_vec * 1), r2);
            _mm256_store_ps(res_ptr.add(floats_per_vec * 2), r3);
            _mm256_store_ps(res_ptr.add(floats_per_vec * 3), r4);

            // Advance pointers
            a_ptr = a_ptr.add(chunk_size);
            b_ptr = b_ptr.add(chunk_size);
            res_ptr = res_ptr.add(chunk_size);
        }
    }

    res
}

#[target_feature(enable = "avx")]
pub unsafe fn simd_add_optimized_stream(a: &[f32], b: &[f32]) -> std::vec::Vec<f32> {
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
    }

    res
}
