use crate::{
    simd::{
        neon::f32x4::{self, F32x4},
        slice::scalar_add,
        SimdLoad, SimdStore,
    },
    utils::alloc_uninit_f32_vec,
    SimdAdd, PARALLEL_CHUNK_SIZE, PARALLEL_SIMD_THRESHOLD,
};
use rayon::prelude::*;

#[target_feature(enable = "neon")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    let mut c =
        alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT).expect("Result Vec allocation error");

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
    let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x4::load(b, f32x4::LANE_COUNT) };
    (a_chunk_simd + b_chunk_simd).store_at(c);
}

#[inline(always)]
fn simd_add_partial_block(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x4::load_partial(b, size) };
    unsafe { (a_chunk_simd + b_chunk_simd).store_at_partial(c) };
}

#[target_feature(enable = "neon")]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(),"Input slices must have the same length for parallel SIMD operations: a.len()={}, b.len()={}", a.len(), b.len());

    // Early return for empty arrays
    if a.is_empty() {
        return Vec::new();
    }

    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() < PARALLEL_SIMD_THRESHOLD {
        return simd_add(a, b);
    }

    let size = a.len();

    let mut c =
        alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT).expect("Result Vec allocation error");

    let step = f32x4::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    c.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_add_block(&a[start_idx + i], &b[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_add_partial_block(
                    &a[start_idx + complete_blocks],
                    &b[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

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
    fn scalar_add(self, rhs: &'b [f32]) -> Self::Output {
        scalar_add(self, rhs)
    }
}
