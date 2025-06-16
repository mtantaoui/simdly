#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) mod float32 {
    use rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSliceMut,
    };

    use crate::simd::{
        avx2::{
            asin::float32::_mm256_asin_ps,
            f32x8::{self, F32x8},
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
    pub(crate) unsafe fn _mm256_acos_ps(d: __m256) -> __m256 {
        // Define π/2 as a vector.
        let pi_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

        // Calculate asin(d) using our existing implementation.
        let asin_d = _mm256_asin_ps(d);

        // Return π/2 - asin(d).
        _mm256_sub_ps(pi_2, asin_d)
    }

    #[inline(always)]
    fn scalar_acos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.acos()).collect()
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn simd_acos(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT);

        let step = f32x8::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
        unsafe { a_chunk_simd.acos().store_at(c) };
    }

    #[inline(always)]
    fn simd_acos_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
        unsafe { a_chunk_simd.acos().store_at_partial(c) };
    }

    #[target_feature(enable = "avx,avx2,fma")]
    fn parallel_simd_acos(a: &[f32]) -> Vec<f32> {
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
