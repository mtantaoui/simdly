use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
pub(crate) mod float32 {

    use crate::simd::{
        neon::f32x4::{self, F32x4},
        traits::{SimdAbs, SimdVec},
        utils::alloc_uninit_f32_vec,
    };

    use super::*;

    #[inline(always)]
    fn scalar_abs(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        a.iter().map(|x| x.abs()).collect()
    }

    #[target_feature(enable = "neon")]
    fn simd_abs(a: &[f32]) -> Vec<f32> {
        assert!(!a.is_empty(), "Size can't be empty (size zero)");

        let size = a.len();

        let mut c = alloc_uninit_f32_vec(size, f32x4::NEON_ALIGNMENT);

        let step = f32x4::LANE_COUNT;

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
        let a_chunk_simd = unsafe { F32x4::load(a, f32x4::LANE_COUNT) };
        unsafe { a_chunk_simd.abs().store_at(c) };
    }

    #[inline(always)]
    fn simd_abs_partial_block(a: *const f32, c: *mut f32, size: usize) {
        // Assumes lengths are f32x4::LANE_COUNT
        let a_chunk_simd = unsafe { F32x4::load_partial(a, size) };
        unsafe { a_chunk_simd.abs().store_at_partial(c) };
    }

    #[target_feature(enable = "neon")]
    fn parallel_simd_abs(a: &[f32]) -> Vec<f32> {
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
