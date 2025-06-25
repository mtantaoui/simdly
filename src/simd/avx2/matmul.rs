// use rayon::{
//     iter::{IndexedParallelIterator, ParallelIterator},
//     slice::ParallelSliceMut,
// };
use std::cmp::min;

use crate::{
    simd::{
        // avx512::f32x8::{self, F32x8, AVX512_ALIGNMENT},
        avx2::f32x8::{self, F32x8},
        traits::SimdVec,
        utils::alloc_zeroed_f32_vec, // utils::alloc_uninit_f32_vec,
    },
    KC, MC, MR, NC, NR,
};
use std::ptr::copy_nonoverlapping;



// Assuming DivCeil trait is available
trait DivCeil: Sized {
    // Copied for self-containment, use your actual trait
    fn msrv_div_ceil(self, rhs: Self) -> Self;
}
impl DivCeil for usize {
    #[inline]
    fn msrv_div_ceil(self, rhs: Self) -> Self {
        self.div_ceil(rhs)
    }
}

/// Calculates the 1D index for a 2D array element in column-major order.
///
/// # Arguments
///
/// * `i` - Row index.
/// * `j` - Column index.
/// * `ld` - Leading dimension (number of rows for column-major).
///
/// # Returns
///
/// The 1D index in the flat array.
#[inline(always)]
fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

#[inline(always)]
fn kernel_8x6(
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro_panel: &mut [f32],
    mr_eff: usize,
    nr_eff: usize,
    kc: usize,
    ldc: usize,
) {
    debug_assert_eq!(MR, 8);
    debug_assert!(
        mr_eff <= MR,
        "mr_eff ({mr_eff}) must be less than or equal to MR ({MR})"
    );
    debug_assert!(
        nr_eff <= NR,
        "nr_eff ({nr_eff}) must be less than or equal to NR ({NR})"
    );

    // Load C columns into SIMD registers
    let c_col1_ptr = c_micro_panel.as_mut_ptr();
    let c_col2_ptr = unsafe { c_micro_panel.as_mut_ptr().add(ldc) };
    let c_col3_ptr = unsafe { c_micro_panel.as_mut_ptr().add(2 * ldc) };
    let c_col4_ptr = unsafe { c_micro_panel.as_mut_ptr().add(3 * ldc) };
    let c_col5_ptr = unsafe { c_micro_panel.as_mut_ptr().add(4 * ldc) };
    let c_col6_ptr = unsafe { c_micro_panel.as_mut_ptr().add(5 * ldc) };
    // let c_col7_ptr = unsafe { c_micro_panel.as_mut_ptr().add(6 * ldc) };
    // let c_col8_ptr = unsafe { c_micro_panel.as_mut_ptr().add(7 * ldc) };
    // let c_col9_ptr = unsafe { c_micro_panel.as_mut_ptr().add(8 * ldc) };
    // let c_col10_ptr = unsafe { c_micro_panel.as_mut_ptr().add(9 * ldc) };
    // let c_col11_ptr = unsafe { c_micro_panel.as_mut_ptr().add(10 * ldc) };
    // let c_col12_ptr = unsafe { c_micro_panel.as_mut_ptr().add(11 * ldc) };
    // let c_col13_ptr = unsafe { c_micro_panel.as_mut_ptr().add(12 * ldc) };
    // let c_col14_ptr = unsafe { c_micro_panel.as_mut_ptr().add(13 * ldc) };
    // let c_col15_ptr = unsafe { c_micro_panel.as_mut_ptr().add(14 * ldc) };
    // let c_col16_ptr = unsafe { c_micro_panel.as_mut_ptr().add(15 * ldc) };

    let (
        mut c_col1,
        mut c_col2,
        mut c_col3,
        mut c_col4,
        mut c_col5,
        mut c_col6,
        // mut c_col7,
        // mut c_col8,
        // mut c_col9,
        // mut c_col10,
        // mut c_col11,
        // mut c_col12,
        // mut c_col13,
        // mut c_col14,
        // mut c_col15,
        // mut c_col16,
    ) = match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => unsafe {
            (
                F32x8::load(c_col1_ptr, mr_eff),
                F32x8::load(c_col2_ptr, mr_eff),
                F32x8::load(c_col3_ptr, mr_eff),
                F32x8::load(c_col4_ptr, mr_eff),
                F32x8::load(c_col5_ptr, mr_eff),
                F32x8::load(c_col6_ptr, mr_eff),
                // F32x8::load(c_col7_ptr, mr_eff),
                // F32x8::load(c_col8_ptr, mr_eff),
                // F32x8::load(c_col9_ptr, mr_eff),
                // F32x8::load(c_col10_ptr, mr_eff),
                // F32x8::load(c_col11_ptr, mr_eff),
                // F32x8::load(c_col12_ptr, mr_eff),
                // F32x8::load(c_col13_ptr, mr_eff),
                // F32x8::load(c_col14_ptr, mr_eff),
                // F32x8::load(c_col15_ptr, mr_eff),
                // F32x8::load(c_col16_ptr, mr_eff),
            )
        },
        std::cmp::Ordering::Less => unsafe {
            (
                F32x8::load_partial(c_col1_ptr, mr_eff),
                F32x8::load_partial(c_col2_ptr, mr_eff),
                F32x8::load_partial(c_col3_ptr, mr_eff),
                F32x8::load_partial(c_col4_ptr, mr_eff),
                F32x8::load_partial(c_col5_ptr, mr_eff),
                F32x8::load_partial(c_col6_ptr, mr_eff),
                // F32x8::load_partial(c_col7_ptr, mr_eff),
                // F32x8::load_partial(c_col8_ptr, mr_eff),
                // F32x8::load_partial(c_col9_ptr, mr_eff),
                // F32x8::load_partial(c_col10_ptr, mr_eff),
                // F32x8::load_partial(c_col11_ptr, mr_eff),
                // F32x8::load_partial(c_col12_ptr, mr_eff),
                // F32x8::load_partial(c_col13_ptr, mr_eff),
                // F32x8::load_partial(c_col14_ptr, mr_eff),
                // F32x8::load_partial(c_col15_ptr, mr_eff),
                // F32x8::load_partial(c_col16_ptr, mr_eff),
            )
        },
        std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
    };

    let mut b_scalar_splat: F32x8;

    for p in 0..kc {
        // Loop over K dimension
        let a_col = F32x8::new(&a_panel[p * MR..p * MR + MR]);
        // Loop over N dimension (columns of B and C micro-panel)

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR]) };
        c_col1 = unsafe { c_col1.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 1]) };
        c_col2 = unsafe { c_col2.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 2]) };
        c_col3 = unsafe { c_col3.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 3]) };
        c_col4 = unsafe { c_col4.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 4]) };
        c_col5 = unsafe { c_col5.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 5]) };
        c_col6 = unsafe { c_col6.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 6]) };
        // c_col7 = unsafe { c_col7.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 7]) };
        // c_col8 = unsafe { c_col8.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 8]) };
        // c_col9 = unsafe { c_col9.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 9]) };
        // c_col10 = unsafe { c_col10.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 10]) };
        // c_col11 = unsafe { c_col11.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 11]) };
        // c_col12 = unsafe { c_col12.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 12]) };
        // c_col13 = unsafe { c_col13.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 13]) };
        // c_col14 = unsafe { c_col14.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 14]) };
        // c_col15 = unsafe { c_col15.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 15]) };
        // c_col16 = unsafe { c_col16.fmadd(a_col, b_scalar_splat) };
    }

    // Store C columns back
    match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => unsafe {
            c_col1.store_at(c_col1_ptr);
            c_col2.store_at(c_col2_ptr);
            c_col3.store_at(c_col3_ptr);
            c_col4.store_at(c_col4_ptr);
            c_col5.store_at(c_col5_ptr);
            c_col6.store_at(c_col6_ptr);
            // c_col7.store_at(c_col7_ptr);
            // c_col8.store_at(c_col8_ptr);
            // c_col9.store_at(c_col9_ptr);
            // c_col10.store_at(c_col10_ptr);
            // c_col11.store_at(c_col11_ptr);
            // c_col12.store_at(c_col12_ptr);
            // c_col13.store_at(c_col13_ptr);
            // c_col14.store_at(c_col14_ptr);
            // c_col15.store_at(c_col15_ptr);
            // c_col16.store_at(c_col16_ptr);
        },
        std::cmp::Ordering::Less => unsafe {
            c_col1.store_at_partial(c_col1_ptr);
            c_col2.store_at_partial(c_col2_ptr);
            c_col3.store_at_partial(c_col3_ptr);
            c_col4.store_at_partial(c_col4_ptr);
            c_col5.store_at_partial(c_col5_ptr);
            c_col6.store_at_partial(c_col6_ptr);
            // c_col7.store_at_partial(c_col7_ptr);
            // c_col8.store_at_partial(c_col8_ptr);
            // c_col9.store_at_partial(c_col9_ptr);
            // c_col10.store_at_partial(c_col10_ptr);
            // c_col11.store_at_partial(c_col11_ptr);
            // c_col12.store_at_partial(c_col12_ptr);
            // c_col13.store_at_partial(c_col13_ptr);
            // c_col14.store_at_partial(c_col14_ptr);
            // c_col15.store_at_partial(c_col15_ptr);
            // c_col16.store_at_partial(c_col16_ptr);
        },
        std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
    };
}

// --- Panel Packing Functions ---
#[inline(always)]
fn pack_panel_a_into(
    dest_slice: &mut [f32],
    a_panel_source_slice: &[f32],
    mr_effective_in_panel: usize,
    kc_panel: usize,
    m_original_matrix: usize,
) {
    debug_assert_eq!(dest_slice.len(), kc_panel * MR, "Dest A slice len mismatch");
    debug_assert!(mr_effective_in_panel <= MR, "mr_eff_a > MR");
    for p_col_in_panel in 0..kc_panel {
        let source_col_start_offset = p_col_in_panel * m_original_matrix;
        let dest_col_segment_start_offset = p_col_in_panel * MR;
        if mr_effective_in_panel > 0 {
            unsafe {
                let src_ptr = a_panel_source_slice.as_ptr().add(source_col_start_offset);
                let dest_ptr = dest_slice.as_mut_ptr().add(dest_col_segment_start_offset);
                copy_nonoverlapping(src_ptr, dest_ptr, mr_effective_in_panel);
            }
        }
    }
}

#[inline(always)]
fn pack_panel_b(
    dest_slice: &mut [f32],
    b_panel_source_slice: &[f32],
    nr_effective_in_panel: usize,
    kc_panel: usize,
    k_original_matrix: usize,
) {
    debug_assert_eq!(dest_slice.len(), kc_panel * NR, "Dest B slice len mismatch");
    debug_assert!(nr_effective_in_panel <= NR, "nr_eff_b > NR");
    for p_row_in_panel in 0..kc_panel {
        let dest_row_start_offset = p_row_in_panel * NR;
        for j_col_in_panel in 0..nr_effective_in_panel {
            let source_index = at(p_row_in_panel, j_col_in_panel, k_original_matrix);
            dest_slice[dest_row_start_offset + j_col_in_panel] = b_panel_source_slice[source_index];
        }
    }
}

// --- Block Packing Functions ---
#[inline(always)]
fn pack_block_a(
    a_block_source_slice: &[f32],
    mc_block: usize,
    kc_block: usize,
    m_original_matrix: usize,
) -> Vec<f32> {
    if mc_block == 0 || kc_block == 0 {
        return Vec::new();
    }
    let num_row_panels = mc_block.msrv_div_ceil(MR);
    let total_packed_size = num_row_panels * kc_block * MR;
    let mut packed_block_a_data = alloc_zeroed_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT);
    let mut current_write_offset_in_packed = 0;
    for i_row_panel_start_in_block in (0..mc_block).step_by(MR) {
        let mr_effective_for_panel = min(MR, mc_block - i_row_panel_start_in_block);
        let panel_source_slice = &a_block_source_slice[i_row_panel_start_in_block..];
        let dest_slice_len_for_this_panel = kc_block * MR;
        let dest_sub_slice = &mut packed_block_a_data[current_write_offset_in_packed
            ..current_write_offset_in_packed + dest_slice_len_for_this_panel];
        pack_panel_a_into(
            dest_sub_slice,
            panel_source_slice,
            mr_effective_for_panel,
            kc_block,
            m_original_matrix,
        );
        current_write_offset_in_packed += dest_slice_len_for_this_panel;
    }
    packed_block_a_data
}

// --- Macro Kernels for GEMMFIP (with fix applied) ---
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macro_kernel_standard(
    block_a_packed: &[f32],
    block_b_already_packed: &[f32],
    c_jc_block_slice: &mut [f32],
    m_orig_c: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    block_b_already_packed
        .chunks(kc_eff_common * NR)
        .enumerate()
        .for_each(|(jr_idx, b_panel_packed)| {
            block_a_packed
                .chunks(MR * kc_eff_common)
                .enumerate()
                .for_each(|(ir_idx, a_panel_packed)| {
                    let nr_eff_micropanel = min(NR, nc_eff_b_block - jr_idx * NR);
                    let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);

                    let micro_panel_start_row_in_block = ic_offset_in_c + ir_idx * MR;
                    let micro_panel_start_col_in_block = jr_idx * NR;
                    let micropanel_start_offset_in_c_block = at(
                        micro_panel_start_row_in_block,
                        micro_panel_start_col_in_block,
                        m_orig_c,
                    );

                    let slice_end_offset_exclusive =
                        if mr_eff_micropanel == 0 || nr_eff_micropanel == 0 {
                            micropanel_start_offset_in_c_block
                        } else {
                            micropanel_start_offset_in_c_block
                                + (nr_eff_micropanel - 1) * m_orig_c
                                + mr_eff_micropanel
                        };

                    let final_slice_end_exclusive =
                        slice_end_offset_exclusive.min(c_jc_block_slice.len());
                    let c_micropanel_for_kernel = &mut c_jc_block_slice
                        [micropanel_start_offset_in_c_block..final_slice_end_exclusive];

                    kernel_8x6(
                        a_panel_packed,
                        b_panel_packed,
                        c_micropanel_for_kernel,
                        mr_eff_micropanel,
                        nr_eff_micropanel,
                        kc_eff_common,
                        m_orig_c,
                    );
                });
        });
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macro_kernel_fused_b(
    block_a_packed: &[f32],
    b_block_original_data_slice: &[f32],
    block_b_packed_dest: &mut [f32],
    c_jc_block_slice: &mut [f32],
    m_orig_c: usize,
    k_orig_b: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    let num_col_panels_in_b_block = nc_eff_b_block.msrv_div_ceil(NR);
    for jr_idx in 0..num_col_panels_in_b_block {
        let nr_eff_this_b_panel = min(NR, nc_eff_b_block - jr_idx * NR);
        let b_panel_dest_offset_in_storage = jr_idx * (kc_eff_common * NR);
        let b_panel_packed_slice_mut = &mut block_b_packed_dest
            [b_panel_dest_offset_in_storage..b_panel_dest_offset_in_storage + kc_eff_common * NR];
        let current_b_panel_source_offset = jr_idx * NR * k_orig_b;
        let current_b_panel_source_slice =
            &b_block_original_data_slice[current_b_panel_source_offset..];
        pack_panel_b(
            b_panel_packed_slice_mut,
            current_b_panel_source_slice,
            nr_eff_this_b_panel,
            kc_eff_common,
            k_orig_b,
        );

        block_a_packed
            .chunks(MR * kc_eff_common)
            .enumerate()
            .for_each(|(ir_idx, a_panel_packed)| {
                let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);

                let micro_panel_start_row_in_block = ic_offset_in_c + ir_idx * MR;
                let micro_panel_start_col_in_block = jr_idx * NR; // jr_idx is from the outer loop here
                let micropanel_start_offset_in_c_block = at(
                    micro_panel_start_row_in_block,
                    micro_panel_start_col_in_block,
                    m_orig_c,
                );
                let slice_end_offset_exclusive = 
                // nr_eff_this_b_panel is the nr_eff for the current micro_panel
                if mr_eff_micropanel == 0 || nr_eff_this_b_panel == 0 {
                     micropanel_start_offset_in_c_block
                } else {
                    micropanel_start_offset_in_c_block
                        + (nr_eff_this_b_panel - 1) * m_orig_c
                        + mr_eff_micropanel
                };

                let final_slice_end_exclusive =
                    slice_end_offset_exclusive.min(c_jc_block_slice.len());
                let c_micropanel_for_kernel = &mut c_jc_block_slice
                    [micropanel_start_offset_in_c_block..final_slice_end_exclusive];

                kernel_8x6(
                    a_panel_packed,
                    b_panel_packed_slice_mut,
                    c_micropanel_for_kernel,
                    mr_eff_micropanel,
                    nr_eff_this_b_panel,
                    kc_eff_common,
                    m_orig_c,
                );
            });
    }
}

// --- Main Matmul Function ---
/// .
/// 
/// # Safety
/// 
/// .
#[target_feature(enable = "avx2,avx,fma")]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 {
        return;
    }
    for jc_start in (0..n).step_by(NC) {
        let nc_eff_block = min(NC, n - jc_start);
        for pc_start in (0..k).step_by(KC) {
            let kc_eff_block = min(KC, k - pc_start);
            if kc_eff_block == 0 {
                continue;
            }
            let b_block_original_data_offset = at(pc_start, jc_start, k);
            let b_block_original_data_slice = &b[b_block_original_data_offset..];
            let num_col_panels_in_b_block = nc_eff_block.msrv_div_ceil(NR);
            let packed_b_storage_size = num_col_panels_in_b_block * kc_eff_block * NR;
            let mut packed_b_storage =
                alloc_zeroed_f32_vec(packed_b_storage_size, f32x8::AVX_ALIGNMENT);
            let mut b_block_has_been_packed_this_jc_pc = false;

            for ic_start in (0..m).step_by(MC) {
                let mc_eff_block = min(MC, m - ic_start);
                if mc_eff_block == 0 {
                    continue;
                }
                let a_block_original_data_offset = at(ic_start, pc_start, m);
                let a_block_original_data_slice = &a[a_block_original_data_offset..];
                let block_a_packed =
                    pack_block_a(a_block_original_data_slice, mc_eff_block, kc_eff_block, m);

                let c_jc_block_slice_start = at(0, jc_start, m);
                // The end index for the slice needs to be based on the number of elements,
                // not just at(0, jc_start + nc_eff_block, m) if nc_eff_block is 0.
                // If nc_eff_block is 0, this loop for jc_start shouldn't run or nc_eff_block=0 handled.
                // It's m rows * nc_eff_block columns. So total elements = m * nc_eff_block from start.
                let c_jc_block_slice_end = c_jc_block_slice_start + m * nc_eff_block;
                let c_target_jc_block_slice = &mut c[c_jc_block_slice_start..c_jc_block_slice_end];

                if !b_block_has_been_packed_this_jc_pc {
                    macro_kernel_fused_b(
                        &block_a_packed,
                        b_block_original_data_slice,
                        &mut packed_b_storage,
                        c_target_jc_block_slice,
                        m,
                        k,
                        mc_eff_block,
                        nc_eff_block,
                        kc_eff_block,
                        ic_start,
                    );
                    if mc_eff_block > 0 && kc_eff_block > 0 {
                        b_block_has_been_packed_this_jc_pc = true;
                    }
                } else {
                    macro_kernel_standard(
                        &block_a_packed,
                        &packed_b_storage,
                        c_target_jc_block_slice,
                        m,
                        mc_eff_block,
                        nc_eff_block,
                        kc_eff_block,
                        ic_start,
                    );
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*; // To access functions and constants from parent module

    // Naive matrix multiplication for result verification (C = A * B)
    // A (m x k), B (k x n), C (m x n)
    // All inputs are flat slices in column-major
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0; m * n];

        let ld_a = m;
        let ld_b = k;
        let ld_c = m;

        for j in 0..n {
            for p in 0..k {
                for i in 0..m {
                    c[at(i, j, ld_c)] = c[at(i, j, ld_c)] + a[at(i, p, ld_a)] * b[at(p, j, ld_b)];
                }
            }
        }

        c
    }

    #[test]
    fn test_at() {
        // For a 3x2 matrix (m=3, n=2), ld=3
        // 0 3
        // 1 4
        // 2 5
        assert_eq!(at(0, 0, 3), 0); // (0,0)
        assert_eq!(at(1, 0, 3), 1); // (1,0)
        assert_eq!(at(0, 1, 3), 3); // (0,1)
        assert_eq!(at(2, 1, 3), 5); // (2,1)
    }

    #[allow(clippy::modulo_one)]
    fn run_matmul_test(m: usize, n: usize, k: usize, use_par: bool) {
        // Override global consts for specific test scenarios if needed,
        // or ensure m,n,k test various relations to MC,NC,KC,MR,NR.
        // For these tests, we use the globally defined MR, NR, etc.

        let a_data: Vec<f32> = (0..(m * k)).map(|x| (x % 100) as f32 / 10.0).collect();
        let b_data: Vec<f32> = (0..(k * n))
            .map(|x| ((x + 50) % 100) as f32 / 10.0)
            .collect();
        let mut c_data = alloc_zeroed_f32_vec(m * n, f32x8::AVX_ALIGNMENT);

        let expected_c = naive_matmul(&a_data, &b_data, m, n, k);

        if use_par {
            unsafe { matmul(&a_data, &b_data, &mut c_data, m, n, k) };
        } else {
            // To test matmul with the *fixed* kernel, you'd need to modify matmul
            // to call kernel_MRxNR_fixed. For now, testing with original kernel_8x1.
            unsafe { matmul(&a_data, &b_data, &mut c_data, m, n, k) };
        }

        for i in 0..(m * n) {
            // If using the original kernel_8x1 and NR > 1, results will be wrong for columns j%NR != 0.
            // This assertion will likely fail for many elements if NR > 1 due to the kernel bug.
            // If NR=1 in constants, it might pass.
            // If the kernel used in matmul/par_matmul is fixed, this should pass.
            // For now, let's assume we want to see it fail with buggy kernel if NR > 1.
            if NR > 1 && (i / m) % NR != 0 && crate::NR > 1 { // (i/m) is column index
                 // If the original buggy kernel is used, columns other than the first in an NR-block won't be computed.
                 // The naive result WILL have values there. The actual C will have zeros (or initial values).
                 // This comparison is tricky. Let's compare against what the BUGGY kernel would produce.
                 // This means the test validates the *current code*, bug and all.
                 // A better test would be against a correctly computed C and expect matmul to match if kernel is fixed.
            }

            assert!(
                (c_data[i] - expected_c[i]).abs() < 1e-1,
                "C[{}] mismatch: got {}, expected {}. (m={}, n={}, k={}, par={})",
                i,
                c_data[i],
                expected_c[i],
                m,
                n,
                k,
                use_par
            );
        }
    }

    // Test cases for matmul and par_matmul
    // Note: These tests will likely FAIL with the provided kernel_8x1 if NR > 1
    // because the kernel only computes the first column of each NR-wide micro-panel.
    // To make them pass, either:
    // 1. Set global const NR = 1.
    // 2. Modify matmul/par_matmul to call the `kernel_MRxNR_fixed`.
    // The tests below are written to compare against a fully correct naive_matmul,
    // so they will expose the kernel_8x1 bug when NR > 1.

    // Small, exact fit for MR, NR potentially
    #[test]
    fn test_matmul_small_exact() {
        run_matmul_test(MR, NR, KC / 2, false);
    }
    #[test]
    fn test_par_matmul_small_exact() {
        run_matmul_test(MR, NR, KC / 2, true);
    }

    // Dimensions smaller than MR, NR
    #[test]
    fn test_matmul_tiny() {
        run_matmul_test(MR / 2, NR / 2, KC / 4, false);
    }
    #[test]
    fn test_par_matmul_tiny() {
        run_matmul_test(MR / 2, NR / 2, KC / 4, true);
    }

    // Dimensions not multiples of MR, NR (test padding)
    #[test]
    fn test_matmul_padding() {
        run_matmul_test(MR + 1, NR + 1, KC / 2 + 1, false);
    }
    #[test]
    fn test_par_matmul_padding() {
        run_matmul_test(MR + 1, NR + 1, KC / 2 + 1, true);
    }

    // Dimensions larger than one block (MC, NC, KC)
    #[test]
    fn test_matmul_large() {
        run_matmul_test(MC + MR, NC + NR, KC + KC / 2, false);
    }
    #[test]
    fn test_par_matmul_large() {
        run_matmul_test(MC + MR, NC + NR, KC + KC / 2, true);
    }

    // Non-square
    #[test]
    fn test_matmul_nonsquare1() {
        run_matmul_test(MC / 2, NC + 5, KC, false);
    }
    #[test]
    fn test_par_matmul_nonsquare1() {
        run_matmul_test(MC / 2, NC + 5, KC, true);
    }
    #[test]
    fn test_matmul_nonsquare2() {
        run_matmul_test(MC + 5, NC / 2, KC, false);
    }
    #[test]
    fn test_par_matmul_nonsquare2() {
        run_matmul_test(MC + 5, NC / 2, KC, true);
    }

    #[allow(clippy::modulo_one)]
    #[test]
    fn test_identity_multiplication() {
        let m = 4;
        let n = 4;
        let k = 4;
        let mut a_data = vec![0.0; m * k]; // Identity for A
        for i in 0..m {
            a_data[at(i, i, m)] = 1.0;
        }

        let b_data: Vec<f32> = (0..(k * n)).map(|x| x as f32).collect();
        let mut c_data = alloc_zeroed_f32_vec(m * n, f32x8::AVX_ALIGNMENT);

        // Using original matmul with potentially buggy kernel
        unsafe { matmul(&a_data, &b_data, &mut c_data, m, n, k) };

        // If kernel is buggy and NR > 1, c_data won't be equal to b_data.
        // For identity A, C should be B.
        // This test will fail if NR > 1 and kernel is not fixed.
        // E.g. if NR=4, m=4. Column 0 of C is computed. Column 1,2,3 of C are not.
        let mut expected_c_for_buggy_kernel = vec![0.0; m * n];
        let naive_c = naive_matmul(&a_data, &b_data, m, n, k); // This is just b_data

        for j_col in 0..n {
            // Only first col in NR-block is computed by buggy kernel
            for i_row in 0..m {
                expected_c_for_buggy_kernel[at(i_row, j_col, m)] = naive_c[at(i_row, j_col, m)];
            }
        }
        for i in 0..(m * n) {
            assert!(
                (c_data[i] - expected_c_for_buggy_kernel[i]).abs() < 1e-3,
                "Identity C[{}] mismatch: got {}, expected_for_buggy_kernel {}. (m={}, n={}, k={})",
                i,
                c_data[i],
                expected_c_for_buggy_kernel[i],
                m,
                n,
                k
            );
        }
    }
}
