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
        avx512::f32x16::{self, F32x16},
        traits::{SimdAbs, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    use super::*;

    /// Computes the absolute value of 16 packed f32 values using a dedicated AVX-512 intrinsic.
    ///
    /// This is the recommended approach for its clarity and directness.
    // This function requires both AVX-512F and the Vector Length (VL) extensions.
    #[inline(always)]
    pub(crate) unsafe fn _mm512_abs_ps(f: __m512) -> __m512 {
        // The `_mm512_abs_ps` intrinsic directly computes the absolute value.
        // The compiler will generate the most efficient instruction, which is often
        // the same bitmasking operation under the hood.
        core::arch::x86_64::_mm512_abs_ps(f)
    }

    #[inline(always)]
    fn scalar_abs(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.abs()).collect()
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn simd_abs(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x16::AVX512_ALIGNMENT);

        let step = f32x16::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        for i in (0..nb_lanes).step_by(step) {
            simd_abs_block(&a[i], &mut c[i]);
        }

        if rem_lanes > 0 {
            simd_abs_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    #[inline(always)]
    fn simd_abs_block(a: *const f32, c: *mut f32) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x16::load(a, f32x16::LANE_COUNT) };
        unsafe { a_chunk_simd.abs().store_at(c) };
    }

    #[inline(always)]
    fn simd_abs_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x16::load_partial(a, size) };
        unsafe { a_chunk_simd.abs().store_at_partial(c) };
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    fn parallel_simd_abs(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x16::AVX512_ALIGNMENT);

        let step = f32x16::LANE_COUNT;

        let nb_lanes = size - (size % step);
        let rem_lanes = size - nb_lanes;

        // Use chunks_exact_mut to ensure we process full blocks of size `step`
        // and handle the remaining elements separately.
        c.par_chunks_exact_mut(step)
            .enumerate()
            .for_each(|(i, c_chunk)| {
                simd_abs_block(&a[i * step], &mut c_chunk[0]);
            });

        // Handle the remaining elements that do not fit into a full block
        if rem_lanes > 0 {
            simd_abs_partial_block(
                &a[nb_lanes],
                &mut c[nb_lanes],
                rem_lanes, // number of reminaing uncomplete lanes
            );
        }

        c
    }

    impl SimdAbs<&[f32]> for &[f32] {
        type Output = Vec<f32>;

        #[inline(always)]
        fn simd_abs(self) -> Self::Output {
            unsafe { simd_abs(self) }
        }

        #[inline(always)]
        fn par_simd_abs(self) -> Self::Output {
            unsafe { parallel_simd_abs(self) }
        }

        #[inline(always)]
        fn scalar_abs(self) -> Self::Output {
            scalar_abs(self)
        }
    }
}
