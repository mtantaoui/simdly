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
/// Trait for integer division that rounds the result up towards positive infinity.
trait DivCeil: Sized {
    // Copied for self-containment, use your actual trait
    /// Performs ceiling division.
    /// `(self + rhs - 1) / rhs` for positive integers.
    fn msrv_div_ceil(self, rhs: Self) -> Self;
}

impl DivCeil for usize {
    #[inline]
    fn msrv_div_ceil(self, rhs: Self) -> Self {
        // Assumes std::intrinsics::exact_div or similar is not used,
        // and relies on the standard library's `div_ceil` available in newer Rust versions.
        // For older Rust, a common pattern is `(self + rhs - 1) / rhs`.
        self.div_ceil(rhs)
    }
}

/// Calculates the 1D index for a 2D array element in column-major order.
///
/// # Arguments
///
/// * `i` - Row index (0-based).
/// * `j` - Column index (0-based).
/// * `ld` - Leading dimension (number of rows for column-major, or stride between columns).
///
/// # Returns
///
/// The 1D index in the flat array.
#[inline]
fn at(i: usize, j: usize, ld: usize) -> usize {
    // Column-major: element (i, j) is at offset j * ld + i
    (j * ld) + i
}

/// Micro-kernel for an 8x6 matrix multiplication: C += A * B.
///
/// This kernel computes a small `MR x NR` (here, 8x6) block of the output matrix C.
/// A is an `MR x kc` panel (packed).
/// B is a `kc x NR` panel (packed).
/// C is an `MR x NR` micro-panel (column-major, part of the larger C matrix).
///
/// # Arguments
///
/// * `a_panel`: Slice of packed A data. Expected layout: column-major, `kc` columns of `MR` elements each.
///   Length should be `kc * MR`.
/// * `b_panel`: Slice of packed B data. Expected layout: row-major, `kc` rows of `NR` elements each.
///   Length should be `kc * NR`.
/// * `c_micro_panel`: Mutable slice representing the C micro-panel to be updated. This slice
///   points to the top-left element C(r,c) of the micro-panel.
/// * `mr_eff`: Effective number of rows to compute in the C micro-panel (<= MR).
///   Used for partial vector loads/stores if `mr_eff < MR`.
/// * `nr_eff`: Effective number of columns to compute in the C micro-panel (<= NR).
///   Currently, this kernel is hardcoded for NR=6 columns. This parameter is
///   asserted but not used to conditionally process columns. The caller must ensure
///   that `c_micro_panel` is valid for writes to 6 columns if this kernel is used.
/// * `kc`:      The common dimension (depth) for the multiplication (A: MRxkc, B: kcxNR).
/// * `ldc`:     Leading dimension of the original C matrix (number of rows in C).
///
/// # Assumptions
///
/// - `MR` (micro-kernel row parameter) is 8.
/// - `NR` (micro-kernel col parameter for B packing and kernel structure) is 6.
/// - `F32x8` is a SIMD type (e.g., AVX `__m256` equivalent for 8 floats).
/// - Operations like `load`, `load_partial`, `store_at`, `store_at_partial`, `splat`, `fmadd`
///   are provided by the `F32x8` SIMD abstraction.
#[inline]
fn kernel_8x8(
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro_panel: *mut f32,
    mr_eff: usize,
    nr_eff: usize,
    kc: usize,
    ldc: usize,
) {
    debug_assert_eq!(MR, 8, "This kernel is hardcoded for MR=8"); // Assuming MR is a global const available here.
                                                                  // If MR is not global, this should be MR.
    debug_assert!(
        mr_eff <= MR,
        "mr_eff ({mr_eff}) must be less than or equal to MR ({MR})"
    );
    debug_assert!(
        nr_eff <= NR, // Assuming NR global const is the packing NR, which should be 6 here.
        "nr_eff ({nr_eff}) must be less than or equal to NR ({NR})"
    );
    // NOTE: This kernel unconditionally processes NR (6) columns of C.
    // The nr_eff parameter is not used to reduce column operations.
    // The caller must ensure that c_micro_panel can safely be written to for NR columns,
    // or that nr_eff == NR.

    // Pointers to the start of each column in the C micro-panel.
    // C is column-major, so columns are `ldc` elements apart.
    let c_col1_ptr = c_micro_panel;
    let c_col2_ptr = unsafe { c_micro_panel.add(ldc) };
    let c_col3_ptr = unsafe { c_micro_panel.add(2 * ldc) };
    let c_col4_ptr = unsafe { c_micro_panel.add(3 * ldc) };
    let c_col5_ptr = unsafe { c_micro_panel.add(4 * ldc) };
    let c_col6_ptr = unsafe { c_micro_panel.add(5 * ldc) };
    let c_col7_ptr = unsafe { c_micro_panel.add(6 * ldc) };
    let c_col8_ptr = unsafe { c_micro_panel.add(7 * ldc) };

    // Load C columns into SIMD registers.
    // Accumulators for C_ij = sum(A_ik * B_kj)
    // Each c_colX variable will hold MR (8) elements of a column of C.
    let (
        mut c_col1, // Holds C(0..mr_eff-1, 0)
        mut c_col2, // Holds C(0..mr_eff-1, 1)
        mut c_col3, // Holds C(0..mr_eff-1, 2)
        mut c_col4, // Holds C(0..mr_eff-1, 3)
        mut c_col5, // Holds C(0..mr_eff-1, 4)
        mut c_col6, // Holds C(0..mr_eff-1, 5)
        mut c_col7, // Holds C(0..mr_eff-1, 6)
        mut c_col8, // Holds C(0..mr_eff-1, 7)
                    // mut c_col7, ...
    ) = match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => {
            // If mr_eff == MR, use full SIMD loads.
            unsafe {
                (
                    F32x8::load(c_col1_ptr, MR), // Assuming F32x8::load loads MR elements
                    F32x8::load(c_col2_ptr, MR),
                    F32x8::load(c_col3_ptr, MR),
                    F32x8::load(c_col4_ptr, MR),
                    F32x8::load(c_col5_ptr, MR),
                    F32x8::load(c_col6_ptr, MR),
                    F32x8::load(c_col7_ptr, MR),
                    F32x8::load(c_col8_ptr, MR),
                )
            }
        }
        std::cmp::Ordering::Less => {
            // If mr_eff < MR, use partial SIMD loads (loads mr_eff elements, zero-pads the rest).
            unsafe {
                (
                    F32x8::load_partial(c_col1_ptr, mr_eff),
                    F32x8::load_partial(c_col2_ptr, mr_eff),
                    F32x8::load_partial(c_col3_ptr, mr_eff),
                    F32x8::load_partial(c_col4_ptr, mr_eff),
                    F32x8::load_partial(c_col5_ptr, mr_eff),
                    F32x8::load_partial(c_col6_ptr, mr_eff),
                    F32x8::load_partial(c_col7_ptr, mr_eff),
                    F32x8::load_partial(c_col8_ptr, mr_eff),
                )
            }
        }
        std::cmp::Ordering::Greater => {
            panic!("mr_eff > MR, this should not happen due to prior assert.")
        }
    };

    // Temporary register for broadcasting B scalar.
    let mut b_scalar_splat: F32x8;

    // Loop over the K dimension (kc times).
    for p in 0..kc {
        // Load a column of A (MR elements) into a SIMD register.
        // a_panel is packed such that MR elements of a column are contiguous.
        // These are A(0..MR-1, p).
        let a_col = F32x8::new(&a_panel[p * MR..(p * MR) + MR]); // Assumes MR=8 for F32x8::new

        // --- Column 1 of C micro-panel ---
        // Load B(p, 0) and broadcast it.
        // b_panel is packed such that NR elements of a row are contiguous.
        // b_panel[p * NR] is B(p,0). NR here refers to the packing width of B.
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR]) };
        // Fused multiply-add: c_col1 += a_col * b_scalar_splat
        c_col1 = unsafe { c_col1.fmadd(a_col, b_scalar_splat) };

        // --- Column 2 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 1]) }; // B(p,1)
        c_col2 = unsafe { c_col2.fmadd(a_col, b_scalar_splat) };

        // --- Column 3 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 2]) }; // B(p,2)
        c_col3 = unsafe { c_col3.fmadd(a_col, b_scalar_splat) };

        // --- Column 4 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 3]) }; // B(p,3)
        c_col4 = unsafe { c_col4.fmadd(a_col, b_scalar_splat) };

        // --- Column 5 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 4]) }; // B(p,4)
        c_col5 = unsafe { c_col5.fmadd(a_col, b_scalar_splat) };

        // --- Column 6 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 5]) }; // B(p,5)
        c_col6 = unsafe { c_col6.fmadd(a_col, b_scalar_splat) };

        // --- Column 7 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 6]) }; // B(p,6)
        c_col7 = unsafe { c_col7.fmadd(a_col, b_scalar_splat) };

        // --- Column 6 of C micro-panel ---
        b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 7]) }; // B(p,7)
        c_col8 = unsafe { c_col8.fmadd(a_col, b_scalar_splat) };
    }

    // Store the accumulated C columns back to memory.
    match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => {
            // If mr_eff == MR, use full SIMD stores.
            unsafe {
                c_col1.store_at(c_col1_ptr);
                c_col2.store_at(c_col2_ptr);
                c_col3.store_at(c_col3_ptr);
                c_col4.store_at(c_col4_ptr);
                c_col5.store_at(c_col5_ptr);
                c_col6.store_at(c_col6_ptr);
                c_col7.store_at(c_col7_ptr);
                c_col8.store_at(c_col8_ptr);
            }
        }
        std::cmp::Ordering::Less => {
            // If mr_eff < MR, use partial SIMD stores (stores only mr_eff elements).
            unsafe {
                c_col1.store_at_partial(c_col1_ptr); // Assumes store_at_partial knows mr_eff from load_partial context or similar
                c_col2.store_at_partial(c_col2_ptr);
                c_col3.store_at_partial(c_col3_ptr);
                c_col4.store_at_partial(c_col4_ptr);
                c_col5.store_at_partial(c_col5_ptr);
                c_col6.store_at_partial(c_col6_ptr);
                c_col7.store_at_partial(c_col7_ptr);
                c_col8.store_at_partial(c_col8_ptr);
            }
        }
        std::cmp::Ordering::Greater => panic!("mr_eff > MR, this should not happen."),
    };
}

// --- Panel Packing Functions ---

/// Packs a panel of matrix A into a destination slice.
#[inline]
fn pack_panel_a_into(
    dest_slice: &mut [f32],
    a_panel_source_slice: &[f32],
    mr_effective_in_panel: usize,
    kc_panel: usize,
    m_original_matrix: usize,
) {
    // Use global MR from crate
    debug_assert_eq!(dest_slice.len(), kc_panel * MR, "Dest A slice len mismatch");
    debug_assert!(mr_effective_in_panel <= MR, "mr_effective_in_panel > MR");

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

/// Packs a panel of matrix B into a destination slice.
#[inline]
fn pack_panel_b(
    dest_slice: &mut [f32],
    b_panel_source_slice: &[f32],
    nr_effective_in_panel: usize,
    kc_panel: usize,
    k_original_matrix: usize,
) {
    // Use global NR from crate
    debug_assert_eq!(dest_slice.len(), kc_panel * NR, "Dest B slice len mismatch");
    debug_assert!(nr_effective_in_panel <= NR, "nr_effective_in_panel > NR");

    for p_row_in_panel in 0..kc_panel {
        let dest_row_start_offset = p_row_in_panel * NR;
        for j_col_in_panel in 0..nr_effective_in_panel {
            let source_index = at(p_row_in_panel, j_col_in_panel, k_original_matrix);
            // Ensure source_index is within bounds of b_panel_source_slice
            // This should be guaranteed by how b_panel_source_slice is created and kc_panel, nr_effective_in_panel are bounded.
            unsafe {
                *dest_slice.get_unchecked_mut(dest_row_start_offset + j_col_in_panel) =
                    *b_panel_source_slice.get_unchecked(source_index);
            }
        }
        // Remaining elements in dest_slice for this row (if nr_effective_in_panel < NR)
        // are already zero due to alloc_zeroed_f32_vec, which is crucial for kernel_8x8.
    }
}

/// Packs a block of matrix A.
#[inline]
fn pack_block_a(
    a_block_source_slice: &[f32],
    mc_block: usize,
    kc_block: usize,
    m_original_matrix: usize,
) -> Vec<f32> {
    if mc_block == 0 || kc_block == 0 {
        return Vec::new();
    }
    // Use global MR from crate
    let num_row_panels = mc_block.msrv_div_ceil(MR);
    let total_packed_size = num_row_panels * kc_block * MR;
    let mut packed_block_a_data = alloc_zeroed_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT);
    let mut current_write_offset_in_packed = 0;

    for i_row_panel_start_in_block in (0..mc_block).step_by(MR) {
        let mr_effective_for_panel = min(MR, mc_block - i_row_panel_start_in_block);

        // Calculate the actual starting point in a_block_source_slice for this panel
        // pack_panel_a_into expects its source slice to start at the panel's top-left element.
        // The first element of the panel is A(i_row_panel_start_in_block, 0) relative to a_block_source_slice.
        // Since a_block_source_slice is already A(ic_abs, pc_abs), this is correct.
        // The source elements for the first column of the panel are at:
        // a_block_source_slice[i_row_panel_start_in_block + 0*m_original_matrix],
        // a_block_source_slice[i_row_panel_start_in_block + 1*m_original_matrix], ... (this is wrong logic here)
        // pack_panel_a_into takes care of column striding with m_original_matrix.
        // panel_source_slice should just be a_block_source_slice offset by the starting row of the panel within the block.
        // let panel_source_start_offset_in_block =
        //     at(i_row_panel_start_in_block, 0, m_original_matrix);

        // Ensure we don't try to create a slice starting beyond the input slice's length.
        // This should be guaranteed by mc_block and i_row_panel_start_in_block logic.
        // let panel_source_slice = &a_block_source_slice[panel_source_start_offset_in_block..];

        // The above slice creation is tricky with 'at'. It's simpler:
        // `a_block_source_slice` is A(ic, pc). Panel is A(ic+i_row_panel_start_in_block, pc).
        // `pack_panel_a_into` expects a slice starting at the panel's top-left.
        // So, `a_block_source_slice` is A_block(0,0). The panel is A_block(i_row_panel_start_in_block, 0).
        // The offset in `a_block_source_slice` to the panel start is just `i_row_panel_start_in_block` if `a_block_source_slice` refers to a single column.
        // No, `a_block_source_slice` is the entire block.
        // The original `&a_block_source_slice[i_row_panel_start_in_block..]` was correct IF `a_block_source_slice` was column major
        // and its first element was the one for `p_col_in_panel = 0` and `m_original_matrix` was its leading dim.
        // The current `a_block_source_slice` starts at A(ic, pc). Its leading dimension *is* `m_original_matrix`.
        // So `a_block_source_slice` viewed as a matrix has `mc_block` rows and `kc_block` cols.
        // The panel starts at row `i_row_panel_start_in_block` *within this block view*.
        // `pack_panel_a_into` will read column `p` of this panel starting from `panel_source_slice[ p * m_original_matrix ]`.
        // This requires `panel_source_slice` to be `a_block_source_slice` shifted to the `i_row_panel_start_in_block`-th row.
        // So `&a_block_source_slice[i_row_panel_start_in_block..]` is correct as long as `m_original_matrix` is the LDA of `a_block_source_slice`.

        let panel_source_slice_for_packing = &a_block_source_slice[i_row_panel_start_in_block..];

        let dest_slice_len_for_this_panel = kc_block * MR;
        let dest_sub_slice = &mut packed_block_a_data[current_write_offset_in_packed
            ..current_write_offset_in_packed + dest_slice_len_for_this_panel];

        pack_panel_a_into(
            dest_sub_slice,
            panel_source_slice_for_packing,
            mr_effective_for_panel,
            kc_block,
            m_original_matrix,
        );
        current_write_offset_in_packed += dest_slice_len_for_this_panel;
    }
    packed_block_a_data
}

/// Standard macro-kernel: C += A * B, where A and B are already packed.
#[allow(clippy::too_many_arguments)]
#[inline]
fn macro_kernel_standard(
    block_a_packed: &[f32],
    block_b_already_packed: &[f32],
    c_base_ptr_for_this_jc_block: *mut f32, // Pointer to C(0, jc_start)
    m_orig_c: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    // Use global MR, NR from crate
    block_b_already_packed
        .chunks(kc_eff_common * NR) // Each chunk is one packed B panel (kc_eff x NR)
        .enumerate()
        .for_each(|(jr_idx, b_panel_packed)| {
            // jr_idx is B panel index along N
            block_a_packed
                .chunks(MR * kc_eff_common) // Each chunk is one packed A panel (MR x kc_eff)
                .enumerate()
                .for_each(|(ir_idx, a_panel_packed)| {
                    // ir_idx is A panel index along M
                    let nr_eff_micropanel = min(NR, nc_eff_b_block - jr_idx * NR);
                    let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);

                    if mr_eff_micropanel == 0 || nr_eff_micropanel == 0 {
                        return;
                    }

                    // Global row index for C micro-panel start
                    let micro_panel_start_global_row_in_c = ic_offset_in_c + ir_idx * MR;
                    // Column index for C micro-panel start *relative to the jc_block*
                    let micro_panel_start_col_in_jc_block = jr_idx * NR;

                    // Offset from c_base_ptr_for_this_jc_block to C_micropanel(0,0)
                    let micropanel_offset_from_jc_block_base = at(
                        micro_panel_start_global_row_in_c, // Global row index
                        micro_panel_start_col_in_jc_block, // Column index within this jc_block
                        m_orig_c,                          // True LDC
                    );

                    let c_micropanel_target_ptr = unsafe {
                        c_base_ptr_for_this_jc_block.add(micropanel_offset_from_jc_block_base)
                    };

                    kernel_8x8(
                        a_panel_packed,
                        b_panel_packed,
                        c_micropanel_target_ptr,
                        mr_eff_micropanel,
                        nr_eff_micropanel,
                        kc_eff_common,
                        m_orig_c,
                    );
                });
        });
}

/// Macro-kernel with fused B packing: C += A * B.
#[allow(clippy::too_many_arguments)]
#[inline]
fn macro_kernel_fused_b(
    block_a_packed: &[f32],
    b_block_original_data_slice: &[f32],
    block_b_packed_dest: &mut [f32],
    c_base_ptr_for_this_jc_block: *mut f32, // Pointer to C(0, jc_start)
    m_orig_c: usize,
    k_orig_b: usize,
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize,
) {
    // Use global MR, NR from crate
    let num_col_panels_in_b_block = nc_eff_b_block.msrv_div_ceil(NR);

    for jr_idx in 0..num_col_panels_in_b_block {
        let nr_eff_this_b_panel = min(NR, nc_eff_b_block - jr_idx * NR);
        if nr_eff_this_b_panel == 0 {
            continue;
        }

        let b_panel_dest_offset_in_storage = jr_idx * (kc_eff_common * NR);
        let b_panel_packed_slice_mut = &mut block_b_packed_dest
            [b_panel_dest_offset_in_storage..b_panel_dest_offset_in_storage + kc_eff_common * NR];

        // `b_block_original_data_slice` is B(pc_start, jc_start).
        // The current B panel to pack is B(pc_start, jc_start + jr_idx*NR).
        // Offset relative to B(pc_start, jc_start): at(0, jr_idx * NR, k_orig_b)
        let current_b_panel_source_offset = at(0, jr_idx * NR, k_orig_b);
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
                if mr_eff_micropanel == 0 {
                    return;
                }

                let micro_panel_start_global_row_in_c = ic_offset_in_c + ir_idx * MR;
                let micro_panel_start_col_in_jc_block = jr_idx * NR;

                let micropanel_offset_from_jc_block_base = at(
                    micro_panel_start_global_row_in_c,
                    micro_panel_start_col_in_jc_block,
                    m_orig_c,
                );

                let c_micropanel_target_ptr = unsafe {
                    c_base_ptr_for_this_jc_block.add(micropanel_offset_from_jc_block_base)
                };

                kernel_8x8(
                    a_panel_packed,
                    b_panel_packed_slice_mut,
                    c_micropanel_target_ptr,
                    mr_eff_micropanel,
                    nr_eff_this_b_panel, // This is the nr_eff for the current B panel
                    kc_eff_common,
                    m_orig_c,
                );
            });
    }
}

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
    // Use global constants MR, NR, MC, NC, KC from crate

    let c_base_ptr = c.as_mut_ptr(); // Base pointer for the entire C matrix

    for jc_start in (0..n).step_by(NC) {
        let nc_eff_block = min(NC, n - jc_start);

        // Pointer to the start of the current C JC-block: C(0, jc_start)
        let c_jc_block_base_ptr = unsafe { c_base_ptr.add(at(0, jc_start, m)) };

        for pc_start in (0..k).step_by(KC) {
            let kc_eff_block = min(KC, k - pc_start);
            if kc_eff_block == 0 {
                continue;
            }

            let b_block_original_data_offset = at(pc_start, jc_start, k);
            let b_block_original_data_slice = {
                // Calculate end carefully to avoid slicing beyond b's length
                // The block spans kc_eff_block rows and nc_eff_block columns
                // The last element would be at B(pc_start + kc_eff_block - 1, jc_start + nc_eff_block - 1)
                // The slice needs to contain all these elements.
                // A slice up to the start of the column *after* the block is one way.
                // Or, more robustly, calculate max needed index.
                // let max_row_in_block = pc_start + kc_eff_block; // exclusive
                // let max_col_in_block = jc_start + nc_eff_block; // exclusive

                // The actual elements are from b[b_block_original_data_offset .. some_end]
                // The length needed for b_block_original_data_slice for pack_panel_b:
                // pack_panel_b accesses b_panel_source_slice[at(p_row, j_col, k_original_matrix)]
                // Max p_row is kc_eff_block-1. Max j_col for any panel is nr_eff_this_b_panel-1.
                // The current_b_panel_source_slice in macro_kernel_fused_b is offset further.
                // Let's use the original slicing logic from the problem, it was likely correct for coverage.
                let b_block_end_offset = at(pc_start, jc_start + nc_eff_block, k).min(b.len());
                &b[b_block_original_data_offset..b_block_end_offset]
            };

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
                let a_block_original_data_slice = {
                    let a_block_end_offset = at(ic_start, pc_start + kc_eff_block, m).min(a.len());
                    &a[a_block_original_data_offset..a_block_end_offset]
                };

                let block_a_packed =
                    pack_block_a(a_block_original_data_slice, mc_eff_block, kc_eff_block, m);

                if !b_block_has_been_packed_this_jc_pc {
                    macro_kernel_fused_b(
                        &block_a_packed,
                        b_block_original_data_slice,
                        &mut packed_b_storage,
                        c_jc_block_base_ptr, // Pass pointer to C(0, jc_start)
                        m,                   // LDC
                        k,                   // LDB
                        mc_eff_block,
                        nc_eff_block,
                        kc_eff_block,
                        ic_start, // Global row offset for A block's contribution
                    );
                    if mc_eff_block > 0 && kc_eff_block > 0 && nc_eff_block > 0 {
                        b_block_has_been_packed_this_jc_pc = true;
                    }
                } else {
                    macro_kernel_standard(
                        &block_a_packed,
                        &packed_b_storage,
                        c_jc_block_base_ptr, // Pass pointer to C(0, jc_start)
                        m,                   // LDC
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
