use std::{cmp::min, ptr::write_bytes};

use simdly::simd::utils::alloc_zeroed_f32_vec;

use simdly::{
    simd::{
        // avx512::f32x8::{self, F32x8, AVX512_ALIGNMENT},
        avx2::f32x8::{self, F32x8},
        traits::SimdVec,
        utils::alloc_uninit_f32_vec,
    },
    KC, MC, MR, NC, NR,
};
use std::ptr::copy_nonoverlapping;

// --- Core Traits and Helpers ---

trait DivCeil: Sized {
    fn msrv_div_ceil(self, rhs: Self) -> Self;
}

impl DivCeil for usize {
    #[time_graph::instrument]
    #[inline(always)]
    fn msrv_div_ceil(self, rhs: Self) -> Self {
        (self + rhs - 1) / rhs
    }
}

#[time_graph::instrument]
#[inline(always)]
fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

// --- KERNEL GENERATION ---
// This macro generates a specialized micro-kernel for a specific number of columns (NR_EFF).
// By using a macro, we avoid code duplication while allowing the compiler to generate
// highly optimized, unrolled code for each specific case.
macro_rules! generate_kernel {
    ($func_name:ident, $nr_eff:expr) => {
        #[time_graph::instrument]
        #[inline(always)]
        fn $func_name(
            a_panel: &[f32],
            b_panel: &[f32], // This panel is now exactly $nr_eff wide
            c_micro_panel: *mut f32,
            mr_eff: usize,
            kc: usize,
            ldc: usize,
        ) {
            let mut c_cols: [F32x8; NR] = [unsafe { F32x8::splat(0.0) }; NR];

            // Load C columns into SIMD registers.
            // This loop will be completely unrolled by the compiler since $nr_eff is a constant.
            if mr_eff == MR {
                for j in 0..$nr_eff {
                    c_cols[j] = unsafe { F32x8::load(c_micro_panel.add(j * ldc), mr_eff) };
                }
            } else {
                for j in 0..$nr_eff {
                    c_cols[j] = unsafe { F32x8::load_partial(c_micro_panel.add(j * ldc), mr_eff) };
                }
            }

            // --- Main compute loop over K dimension ---
            for p in 0..kc {
                let a_col = F32x8::new(&a_panel[p * MR..]);

                // This inner loop over the columns of B and C is the key.
                // It will be fully unrolled, creating a flat sequence of FMAs.
                for j in 0..$nr_eff {
                    let b_scalar = unsafe { *b_panel.get_unchecked(p * $nr_eff + j) };
                    c_cols[j] = unsafe { c_cols[j].fmadd(a_col, F32x8::splat(b_scalar)) };
                }
            }

            // Store the results back to C.
            if mr_eff == MR {
                for j in 0..$nr_eff {
                    unsafe { c_cols[j].store_at(c_micro_panel.add(j * ldc)) };
                }
            } else {
                for j in 0..$nr_eff {
                    unsafe { c_cols[j].store_at_partial(c_micro_panel.add(j * ldc)) };
                }
            }
        }
    };
}

// Instantiate the entire family of kernels.
generate_kernel!(kernel_8x1, 1);
generate_kernel!(kernel_8x2, 2);
generate_kernel!(kernel_8x3, 3);
generate_kernel!(kernel_8x4, 4);
generate_kernel!(kernel_8x5, 5);
generate_kernel!(kernel_8x6, 6);
generate_kernel!(kernel_8x7, 7);
generate_kernel!(kernel_8x8, 8);

// --- Panel Packing Functions ---

/// Packs a panel of matrix A. This function now explicitly handles zero-padding
/// because the destination buffer is uninitialized.
#[time_graph::instrument]
#[inline(always)]
fn pack_panel_a_into(
    dest_slice: &mut [f32],
    // --- FIX: Take the full 'A' matrix and the panel's starting coordinates ---
    a_full_matrix: &[f32],
    panel_start_row: usize,
    panel_start_col: usize,
    mr_effective: usize,
    kc_panel: usize,
    m_original_matrix: usize,
) {
    let panel_base_offset = at(panel_start_row, panel_start_col, m_original_matrix);
    for p_col_in_panel in 0..kc_panel {
        let dest_col_offset = p_col_in_panel * MR;
        if mr_effective > 0 {
            let src_offset = panel_base_offset + p_col_in_panel * m_original_matrix;
            unsafe {
                let src_ptr = a_full_matrix.as_ptr().add(src_offset);
                let dest_ptr = dest_slice.as_mut_ptr().add(dest_col_offset);
                copy_nonoverlapping(src_ptr, dest_ptr, mr_effective);
            }
        }
        if mr_effective < MR {
            let padding_start = dest_col_offset + mr_effective;
            let padding_len = MR - mr_effective;
            unsafe {
                let dest_ptr = dest_slice.as_mut_ptr().add(padding_start);
                write_bytes(dest_ptr, 0, padding_len);
            }
        }
    }
}

#[time_graph::instrument]
#[inline(always)]
fn pack_panel_b(
    dest_slice: &mut [f32],
    // --- FIX: Take the full 'B' matrix and the panel's starting coordinates ---
    b_full_matrix: &[f32],
    panel_start_row: usize,
    panel_start_col: usize,
    panel_width: usize,
    kc_panel: usize,
    k_original_matrix: usize,
) {
    debug_assert_eq!(dest_slice.len(), kc_panel * panel_width);
    let panel_base_offset = at(panel_start_row, panel_start_col, k_original_matrix);
    let panel_base_ptr = unsafe { b_full_matrix.as_ptr().add(panel_base_offset) };
    let dest_ptr = dest_slice.as_mut_ptr();

    for j_col_in_panel in 0..panel_width {
        let src_col_start_ptr = unsafe { panel_base_ptr.add(j_col_in_panel * k_original_matrix) };
        let mut dest_write_ptr = unsafe { dest_ptr.add(j_col_in_panel) };
        for p_row_in_panel in 0..kc_panel {
            unsafe {
                *dest_write_ptr = *src_col_start_ptr.add(p_row_in_panel);
                dest_write_ptr = dest_write_ptr.add(panel_width);
            }
        }
    }
}

#[time_graph::instrument]
#[inline(always)]
fn pack_block_a(
    // --- FIX: Take the full 'A' matrix and block coordinates ---
    a_full_matrix: &[f32],
    block_start_row: usize,
    block_start_col: usize,
    mc_block: usize,
    kc_block: usize,
    m_original_matrix: usize,
) -> Vec<f32> {
    if mc_block == 0 || kc_block == 0 {
        return Vec::new();
    }

    let num_row_panels = mc_block.msrv_div_ceil(MR);
    let total_packed_size = num_row_panels * kc_block * MR;
    let mut packed_block_a_data = alloc_uninit_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT);

    let mut current_write_offset_in_packed = 0;
    for i_panel_in_block in 0..num_row_panels {
        let panel_start_row_in_block = i_panel_in_block * MR;
        let mr_effective = min(MR, mc_block - panel_start_row_in_block);

        let dest_slice_len = kc_block * MR;
        let dest_sub_slice = &mut packed_block_a_data
            [current_write_offset_in_packed..current_write_offset_in_packed + dest_slice_len];

        pack_panel_a_into(
            dest_sub_slice,
            a_full_matrix,
            block_start_row + panel_start_row_in_block,
            block_start_col,
            mr_effective,
            kc_block,
            m_original_matrix,
        );
        current_write_offset_in_packed += dest_slice_len;
    }
    packed_block_a_data
}

// --- Macro-Kernel Dispatcher ---

/// Macro-kernel with fused B packing. It now correctly uses a write-offset
/// for the B packing buffer.
#[allow(clippy::too_many_arguments)]
#[time_graph::instrument]
#[inline(always)]
fn macro_kernel_fused_b(
    block_a_packed: &[f32],
    b_full_matrix: &[f32],
    b_block_start_row: usize,
    b_block_start_col: usize,
    block_b_packed_dest: &mut [f32],
    c_base_ptr_for_this_jc_block: *mut f32,
    m_orig_c: usize,
    k_orig_b: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    let num_col_panels_in_b_block = nc_eff_b_block.msrv_div_ceil(NR);
    let mut b_packed_offset = 0;

    for jr_idx in 0..num_col_panels_in_b_block {
        let nr_eff_this_b_panel = min(NR, nc_eff_b_block - jr_idx * NR);
        if nr_eff_this_b_panel == 0 {
            continue;
        }

        let b_panel_packed_size = kc_eff_common * nr_eff_this_b_panel;
        let b_panel_packed_slice_mut =
            &mut block_b_packed_dest[b_packed_offset..b_packed_offset + b_panel_packed_size];

        pack_panel_b(
            b_panel_packed_slice_mut,
            b_full_matrix,
            b_block_start_row,
            b_block_start_col + jr_idx * NR,
            nr_eff_this_b_panel,
            kc_eff_common,
            k_orig_b,
        );

        run_computation_for_b_panel(
            block_a_packed,
            b_panel_packed_slice_mut,
            c_base_ptr_for_this_jc_block,
            m_orig_c,
            mc_eff_a_block,
            kc_eff_common,
            ic_offset_in_c,
            jr_idx,
            nr_eff_this_b_panel,
        );
        b_packed_offset += b_panel_packed_size;
    }
}

/// **CRITICAL FIX**: This function is now also a proper dispatcher. It no longer
/// incorrectly assumes all panels are NR-wide.
#[allow(clippy::too_many_arguments)]
#[time_graph::instrument]
#[inline(always)]
fn macro_kernel_standard(
    block_a_packed: &[f32],
    block_b_already_packed: &[f32],
    c_base_ptr_for_this_jc_block: *mut f32,
    m_orig_c: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    let num_col_panels = nc_eff_b_block.msrv_div_ceil(NR);
    let mut b_packed_offset = 0;

    for jr_idx in 0..num_col_panels {
        let nr_eff_this_b_panel = min(NR, nc_eff_b_block - jr_idx * NR);
        if nr_eff_this_b_panel == 0 {
            continue;
        }

        let b_panel_packed_size = kc_eff_common * nr_eff_this_b_panel;
        let b_panel_packed_slice =
            &block_b_already_packed[b_packed_offset..b_packed_offset + b_panel_packed_size];

        run_computation_for_b_panel(
            block_a_packed,
            b_panel_packed_slice,
            c_base_ptr_for_this_jc_block,
            m_orig_c,
            mc_eff_a_block,
            kc_eff_common,
            ic_offset_in_c,
            jr_idx,
            nr_eff_this_b_panel,
        );
        b_packed_offset += b_panel_packed_size;
    }
}

/// Helper function to de-duplicate the computation/dispatch logic from both macro-kernels.
#[allow(clippy::too_many_arguments)]
#[time_graph::instrument]
#[inline(always)]
fn run_computation_for_b_panel(
    block_a_packed: &[f32],
    b_panel_packed_slice: &[f32],
    c_base_ptr_for_this_jc_block: *mut f32,
    m_orig_c: usize,
    mc_eff_a_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
    jr_idx: usize,
    nr_eff_this_b_panel: usize,
) {
    block_a_packed
        .chunks(MR * kc_eff_common)
        .enumerate()
        .for_each(|(ir_idx, a_panel_packed)| {
            let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);
            if mr_eff_micropanel == 0 {
                return;
            }

            let micropanel_offset = at(ic_offset_in_c + ir_idx * MR, jr_idx * NR, m_orig_c);
            let c_micropanel_target_ptr =
                unsafe { c_base_ptr_for_this_jc_block.add(micropanel_offset) };

            // KERNEL DISPATCH
            match nr_eff_this_b_panel {
                8 => kernel_8x8(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                7 => kernel_8x7(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                6 => kernel_8x6(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                5 => kernel_8x5(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                4 => kernel_8x4(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                3 => kernel_8x3(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                2 => kernel_8x2(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                1 => kernel_8x1(
                    a_panel_packed,
                    b_panel_packed_slice,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    kc_eff_common,
                    m_orig_c,
                ),
                _ => {}
            }
        });
}

// --- CORRECTED TOP-LEVEL MATMUL FUNCTION ---

// --- TOP-LEVEL MATMUL FUNCTION (Corrected) ---
#[time_graph::instrument]
#[target_feature(enable = "avx2,avx,fma")]
pub unsafe fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 {
        return;
    }

    (0..n).into_iter().step_by(NC).for_each(|jc_start| {
        let nc_eff_block = min(NC, n - jc_start);

        for pc_start in (0..k).step_by(KC) {
            let kc_eff_block = min(KC, k - pc_start);
            if kc_eff_block == 0 {
                continue;
            }

            // Size calculation for the B packing buffer
            let num_b_panels = nc_eff_block.msrv_div_ceil(NR);
            let last_b_panel_width = if nc_eff_block % NR == 0 {
                NR
            } else {
                nc_eff_block % NR
            };
            let packed_b_storage_size =
                (num_b_panels - 1) * kc_eff_block * NR + kc_eff_block * last_b_panel_width;
            let mut packed_b_storage =
                alloc_uninit_f32_vec(packed_b_storage_size, f32x8::AVX_ALIGNMENT);

            let mut b_block_has_been_packed_this_jc_pc = false;

            for ic_start in (0..m).step_by(MC) {
                let mc_eff_block = min(MC, m - ic_start);
                if mc_eff_block == 0 {
                    continue;
                }

                let c_base_ptr = c.as_mut_ptr();
                let c_jc_block_base_ptr = c_base_ptr.add(at(0, jc_start, m));

                let block_a_packed =
                    pack_block_a(a, ic_start, pc_start, mc_eff_block, kc_eff_block, m);

                if !b_block_has_been_packed_this_jc_pc {
                    // --- FIX: Pass the correct local variables ---
                    macro_kernel_fused_b(
                        &block_a_packed,
                        b,
                        pc_start,
                        jc_start,
                        &mut packed_b_storage,
                        c_jc_block_base_ptr,
                        m,
                        k,
                        mc_eff_block, // Corrected from mc_eff_a_block
                        nc_eff_block,
                        kc_eff_block, // Corrected from kc_eff_common
                        ic_start,     // Corrected from ic_offset_in_c
                    );
                    if mc_eff_block > 0 && kc_eff_block > 0 && nc_eff_block > 0 {
                        b_block_has_been_packed_this_jc_pc = true;
                    }
                } else {
                    // --- FIX: Pass the correct local variables ---
                    macro_kernel_standard(
                        &block_a_packed,
                        &packed_b_storage,
                        c_jc_block_base_ptr,
                        m,
                        mc_eff_block, // Corrected from mc_eff_a_block
                        nc_eff_block, // Corrected from nc_eff_b_block
                        kc_eff_block, // Corrected from kc_eff_common
                        ic_start,     // Corrected from ic_offset_in_c
                    );
                }
            }
        }
    });
}

fn main() {
    let m = 64;
    let n = 64;
    let k = 64;

    let a_data: Vec<f32> = (0..(m * k)).map(|x| (x % 100) as f32 / 10.0).collect();
    let b_data: Vec<f32> = (0..(k * n))
        .map(|x| ((x + 50) % 100) as f32 / 10.0)
        .collect();
    let mut c_data = alloc_zeroed_f32_vec(m * n, f32x8::AVX_ALIGNMENT);

    // Enable performance profiling
    time_graph::enable_data_collection(true);

    unsafe { matmul(&a_data, &b_data, &mut c_data, m, n, k) };

    // Get and print the performance profiling results
    let graph = time_graph::get_full_graph();
    // The following output formats are available but commented out:
    // println!("{}", graph.as_dot()); // DOT format for visualization
    // println!("{}", graph.as_json());     // JSON format
    // println!("{}", graph.as_table());    // Full table
    println!("{}", graph.as_short_table()); // Condensed table
}
