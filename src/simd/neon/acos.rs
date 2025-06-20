use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub(crate) mod float32 {

    use crate::simd::{
        neon::{
            asin::float32::vasinq_f32,
            f32x4::{self, F32x4},
        },
        traits::{SimdAcos, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    use super::*;

    /// Computes `acos(d)` for each element in the input vector `d`.
    ///
    /// # Method
    /// This function leverages the identity `acos(x) = π/2 - asin(x)`.
    /// It calls the optimized `_mm256_asin_ps` function and subtracts the result from π/2.
    /// The `#[inline]` attribute ensures that the compiler will merge these functions,
    /// eliminating any function call overhead and resulting in optimal performance.
    #[inline(always)]
    /// Computes arccos for four f32 lanes with high precision using the identity
    /// acos(x) = π/2 - asin(x).
    pub(crate) unsafe fn vacosq_f32(d: float32x4_t) -> float32x4_t {
        // Calculate asin(d)
        let asin_d = vasinq_f32(d);

        // Get π/2 vector
        let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

        // acos(d) = π/2 - asin(d)

        // The NaN handling for out-of-domain inputs is already done by vasinq_f32.
        // If asin_d is NaN, then pi_over_2 - NaN will also be NaN, which is correct.
        vsubq_f32(pi_over_2, asin_d)
    }

    #[inline(always)]
    fn scalar_acos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.acos()).collect()
    }

    #[target_feature(enable = "neon")]
    fn simd_acos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        for i in (0..nb_lanes).step_by(step) {
            simd_acos_block(&a[i], &mut c[i]);
        }

        if rem_lanes > 0 {
            simd_acos_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    #[inline(always)]
    fn simd_acos_block(a: *const f32, c: *mut f32) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
        unsafe { a_chunk_simd.acos().store_at(c) };
    }

    #[inline(always)]
    fn simd_acos_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
        unsafe { a_chunk_simd.acos().store_at_partial(c) };
    }

    #[target_feature(enable = "neon")]
    fn parallel_simd_acos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        // Use chunks_exact_mut to ensure we process full blocks of size `step`
        // and handle the remaining elements separately.
        c.par_chunks_exact_mut(step)
            .enumerate()
            .for_each(|(i, c_chunk)| {
                simd_acos_block(&a[i * step], &mut c_chunk[0]);
            });

        // Handle the remaining elements that do not fit into a full block
        if rem_lanes > 0 {
            simd_acos_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    impl SimdAcos<&[f32]> for &[f32] {
        type Output = Vec<f32>;

        #[inline(always)]
        fn simd_acos(self) -> Self::Output {
            unsafe { simd_acos(self) }
        }

        #[inline(always)]
        fn par_simd_acos(self) -> Self::Output {
            unsafe { parallel_simd_acos(self) }
        }

        #[inline(always)]
        fn scalar_acos(self) -> Self::Output {
            scalar_acos(self)
        }
    }
}
