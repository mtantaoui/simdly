// use rayon::{
//     iter::{IndexedParallelIterator, ParallelIterator},
//     slice::ParallelSliceMut,
// };
use std::{
    alloc::{alloc, dealloc, Layout},
    cmp::min,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
#[inline(always)]
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
    let c_col1_ptr = c_micro_panel.as_mut_ptr();
    let c_col2_ptr = unsafe { c_micro_panel.as_mut_ptr().add(ldc) };
    let c_col3_ptr = unsafe { c_micro_panel.as_mut_ptr().add(2 * ldc) };
    let c_col4_ptr = unsafe { c_micro_panel.as_mut_ptr().add(3 * ldc) };
    let c_col5_ptr = unsafe { c_micro_panel.as_mut_ptr().add(4 * ldc) };
    let c_col6_ptr = unsafe { c_micro_panel.as_mut_ptr().add(5 * ldc) };

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

        // Commented out: FMAs for more columns if NR was larger
        // b_scalar_splat = unsafe { F32x8::splat(b_panel[p * NR + 6]) };
        // c_col7 = unsafe { c_col7.fmadd(a_col, b_scalar_splat) }; ...
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
            }
        }
        std::cmp::Ordering::Greater => panic!("mr_eff > MR, this should not happen."),
    };
}

// --- Panel Packing Functions ---

/// Packs a panel of matrix A into a destination slice.
///
/// Matrix A is column-major. The packed format is also column-major by MR-sized segments.
/// Specifically, for each column `p` in the source panel, `mr_effective_in_panel` elements
/// are copied to `dest_slice`. If `mr_effective_in_panel < MR`, the remaining elements
/// in the MR-sized destination segment should be zero (achieved by `alloc_zeroed_f32_vec`
/// in the calling `pack_block_a`).
///
/// # Arguments
///
/// * `dest_slice`:            The destination slice for packed data. Length must be `kc_panel * MR`.
/// * `a_panel_source_slice`:  The source slice from matrix A. This slice should start at the
///   first element of the panel to be packed. Assumed to be column-major.
/// * `mr_effective_in_panel`: The number of rows to actually copy for each column segment (<= MR).
/// * `kc_panel`:              The number of columns in this panel (k-dimension).
/// * `m_original_matrix`:     The total number of rows in the original matrix A (leading dimension of A).
#[inline(always)]
fn pack_panel_a_into(
    dest_slice: &mut [f32],
    a_panel_source_slice: &[f32],
    mr_effective_in_panel: usize,
    kc_panel: usize,
    m_original_matrix: usize,
) {
    // MR is assumed to be a global constant (e.g., 8)
    const MR: usize = 8; // Example, should match kernel's MR
    debug_assert_eq!(dest_slice.len(), kc_panel * MR, "Dest A slice len mismatch");
    debug_assert!(mr_effective_in_panel <= MR, "mr_effective_in_panel > MR");

    // Iterate over columns of the panel (p_col_in_panel corresponds to dimension k)
    for p_col_in_panel in 0..kc_panel {
        // Calculate the starting offset in the source slice for the current column.
        // Source is column-major, so columns are `m_original_matrix` elements apart.
        let source_col_start_offset = p_col_in_panel * m_original_matrix;

        // Calculate the starting offset in the destination slice for the current packed column segment.
        // Destination is packed with MR-sized column segments.
        let dest_col_segment_start_offset = p_col_in_panel * MR;

        if mr_effective_in_panel > 0 {
            unsafe {
                // Pointer to the start of the source column
                let src_ptr = a_panel_source_slice.as_ptr().add(source_col_start_offset);
                // Pointer to the start of the destination segment
                let dest_ptr = dest_slice.as_mut_ptr().add(dest_col_segment_start_offset);
                // Copy `mr_effective_in_panel` elements.
                // If mr_effective_in_panel < MR, the rest of the MR-sized segment in dest_slice
                // should already be zero-padded from allocation.
                copy_nonoverlapping(src_ptr, dest_ptr, mr_effective_in_panel);
            }
        }
        // If mr_effective_in_panel < MR, the remaining (MR - mr_effective_in_panel) elements
        // in the destination segment `dest_slice[dest_col_segment_start_offset + mr_effective_in_panel .. dest_col_segment_start_offset + MR]`
        // must be zero for correctness with SIMD loads in the kernel. This is handled by `alloc_zeroed_f32_vec`.
    }
}

/// Packs a panel of matrix B into a destination slice.
///
/// Matrix B is typically column-major. The packed format is row-major by NR-sized segments,
/// suitable for the micro-kernel (where B scalars are broadcast).
/// Specifically, for each row `p` in the source panel (which corresponds to dimension k),
/// `nr_effective_in_panel` elements are copied contiguously into `dest_slice`.
///
/// # Arguments
///
/// * `dest_slice`:              The destination slice for packed data. Length must be `kc_panel * NR`.
/// * `b_panel_source_slice`:    The source slice from matrix B. This slice should start at the
///   first element of the panel to be packed. Assumed to be column-major.
/// * `nr_effective_in_panel`:   The number of columns to actually copy for each row segment (<= NR).
/// * `kc_panel`:                The number of rows in this panel (k-dimension).
/// * `k_original_matrix`:       The total number of rows in the original matrix B (leading dimension of B).
#[inline(always)]
fn pack_panel_b(
    dest_slice: &mut [f32],
    b_panel_source_slice: &[f32],
    nr_effective_in_panel: usize,
    kc_panel: usize,
    k_original_matrix: usize, // This is 'k', the number of rows in original B matrix (ldb)
) {
    // NR is assumed to be a global constant (e.g., 6 for kernel_8x6)
    const NR: usize = 6; // Example, should match kernel's NR
    debug_assert_eq!(dest_slice.len(), kc_panel * NR, "Dest B slice len mismatch");
    debug_assert!(nr_effective_in_panel <= NR, "nr_effective_in_panel > NR");

    // Iterate over rows of the panel (p_row_in_panel corresponds to dimension k)
    for p_row_in_panel in 0..kc_panel {
        // Calculate the starting offset in the destination slice for the current packed row segment.
        // Destination is packed with NR-sized row segments.
        let dest_row_start_offset = p_row_in_panel * NR;

        // Iterate over columns within the effective width of the panel
        for j_col_in_panel in 0..nr_effective_in_panel {
            // Source B is column-major. Element B(p_row_in_panel, j_col_in_panel)
            // is at index `at(p_row_in_panel, j_col_in_panel, k_original_matrix)`.
            let source_index = at(p_row_in_panel, j_col_in_panel, k_original_matrix);
            dest_slice[dest_row_start_offset + j_col_in_panel] = b_panel_source_slice[source_index];
        }
        // If nr_effective_in_panel < NR, the remaining (NR - nr_effective_in_panel) elements
        // in the destination segment `dest_slice[dest_row_start_offset + nr_effective_in_panel .. dest_row_start_offset + NR]`
        // will be zero if `dest_slice` was zero-allocated. This padding is important if the kernel
        // unconditionally reads NR elements from b_panel rows (which it does via `b_panel[p * NR + offset]`).
    }
}

// --- Block Packing Functions ---

/// Packs a block of matrix A.
///
/// This function takes a block (a submatrix) of A and packs it into a contiguous
/// buffer. The packing is done panel by panel, where each panel is `MR x kc_block`.
/// The resulting `packed_block_a_data` is what `macro_kernel_standard` and
/// `macro_kernel_fused_b` expect for `block_a_packed`.
///
/// # Arguments
///
/// * `a_block_source_slice`: Slice representing the block of matrix A to be packed.
///   Assumed to be column-major and starting at A(ic_start, pc_start).
/// * `mc_block`:             Number of rows in this block of A.
/// * `kc_block`:             Number of columns in this block of A (k-dimension).
/// * `m_original_matrix`:    Total number of rows in the original matrix A (leading dimension of A).
///
/// # Returns
///
/// A `Vec<f32>` containing the packed data, zero-padded for alignment and MR dimensions.
/// The layout is a sequence of A-panels, each `MR x kc_block` (packed column-wise internally).
#[inline(always)]
fn pack_block_a(
    a_block_source_slice: &[f32],
    mc_block: usize,
    kc_block: usize,
    m_original_matrix: usize,
) -> Vec<f32> {
    if mc_block == 0 || kc_block == 0 {
        return Vec::new(); // Nothing to pack
    }

    // Calculate how many MR-high row panels are needed to cover mc_block rows.
    let num_row_panels = mc_block.msrv_div_ceil(MR);
    // Each panel is effectively MR rows tall (padded) and kc_block columns wide.
    // The packed storage for one panel is MR * kc_block.
    let total_packed_size = num_row_panels * kc_block * MR;

    // Allocate a zeroed vector for the packed data. Zeroing is important for padding
    // when mr_effective_for_panel < MR.
    let mut packed_block_a_data = alloc_zeroed_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT); // Assuming alloc_zeroed_f32_vec exists

    let mut current_write_offset_in_packed = 0;

    // Iterate over MR-sized row strips (panels) within the A block.
    for i_row_panel_start_in_block in (0..mc_block).step_by(MR) {
        // Effective number of rows for this specific panel (handles edges where mc_block is not a multiple of MR).
        let mr_effective_for_panel = min(MR, mc_block - i_row_panel_start_in_block);

        // The source data for this panel starts at `i_row_panel_start_in_block` rows down
        // from the beginning of `a_block_source_slice`.
        // Note: `a_block_source_slice` itself is already a view into the original A matrix.
        // The `pack_panel_a_into` function expects a source slice starting at the panel's top-left.
        // The indexing within `pack_panel_a_into` (`p_col_in_panel * m_original_matrix`) uses the
        // original matrix's leading dimension relative to the start of `a_panel_source_slice` argument.
        // Here, `a_block_source_slice` is A(ic, pc) to A(ic+mc-1, pc+kc-1).
        // The panel starts at row `i_row_panel_start_in_block` *within this block*.
        // So, the source slice for `pack_panel_a_into` should be relative to `a_block_source_slice`.
        let panel_source_slice = &a_block_source_slice[i_row_panel_start_in_block..];
        // The m_original_matrix parameter to pack_panel_a_into is the LDA of the source from which panel_source_slice is derived,
        // which is indeed m_original_matrix (the LDA of original A), because panel_source_slice still "sees" original A's column stride.

        // Length of the destination sub-slice for this one panel.
        let dest_slice_len_for_this_panel = kc_block * MR;

        // Get a mutable sub-slice of the pre-allocated packed_block_a_data.
        let dest_sub_slice = &mut packed_block_a_data[current_write_offset_in_packed
            ..current_write_offset_in_packed + dest_slice_len_for_this_panel];

        // Pack the current panel.
        pack_panel_a_into(
            dest_sub_slice,
            panel_source_slice, // Source data for this panel, starting at its top-left relative to a_block_source_slice.
            mr_effective_for_panel,
            kc_block,
            m_original_matrix, // Leading dimension of original A, used for striding in source.
        );

        // Advance the write offset for the next panel.
        current_write_offset_in_packed += dest_slice_len_for_this_panel;
    }
    packed_block_a_data
}

// --- Macro Kernels for GEMM ---

/// Standard macro-kernel: C += A * B, where A and B are already packed.
///
/// This function iterates over micro-panels of A and B, calling the micro-kernel
/// `kernel_8x6` for each.
///
/// # Arguments
///
/// * `block_a_packed`:         Packed block of A. Layout: sequence of (MR x kc_eff_common) panels.
/// * `block_b_already_packed`: Packed block of B. Layout: sequence of (kc_eff_common x NR) panels.
/// * `c_jc_block_slice`:       Mutable slice of the C matrix corresponding to the current JC block
///   (all M rows, nc_eff_b_block columns). Updates are made into this slice.
/// * `m_orig_c`:               Leading dimension (number of rows) of the original C matrix.
/// * `mc_eff_a_block`:         Effective number of rows in the A block.
/// * `nc_eff_b_block`:         Effective number of columns in the B block.
/// * `kc_eff_common`:          Effective common dimension (depth k) for this block multiplication.
/// * `ic_offset_in_c`:         Row offset within `c_jc_block_slice` where the current A block's
///   contribution to C should start. This is `ic_start` from `matmul`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macro_kernel_standard(
    block_a_packed: &[f32],
    block_b_already_packed: &[f32],
    c_jc_block_slice: &mut [f32],
    m_orig_c: usize, // Leading dimension of C
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize, // Starting row in C for this A block (ic_start)
) {
    const MR: usize = 8; // Assumed global constant
    const NR: usize = 6; // Assumed global constant, matching kernel_8x6 and B packing

    // Iterate over column panels of B (jr_idx is the panel index along N dimension).
    // Each B panel is (kc_eff_common rows) x (NR columns).
    // The packed B block is a flat sequence of these panels.
    block_b_already_packed
        .chunks(kc_eff_common * NR) // Each chunk is one packed B panel
        .enumerate()
        .for_each(|(jr_idx, b_panel_packed)| {
            // b_panel_packed is a slice representing one (kc_eff_common x NR) panel of B.

            // Iterate over row panels of A (ir_idx is the panel index along M dimension).
            // Each A panel is (MR rows) x (kc_eff_common columns).
            // The packed A block is a flat sequence of these panels.
            block_a_packed
                .chunks(MR * kc_eff_common) // Each chunk is one packed A panel
                .enumerate()
                .for_each(|(ir_idx, a_panel_packed)| {
                    // a_panel_packed is a slice representing one (MR x kc_eff_common) panel of A.

                    // Effective dimensions for the current micro-panel.
                    let nr_eff_micropanel = min(NR, nc_eff_b_block - jr_idx * NR);
                    let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);

                    if mr_eff_micropanel == 0 || nr_eff_micropanel == 0 {
                        return; // Skip if micro-panel is empty.
                    }

                    // Calculate start of the C micro-panel within c_jc_block_slice.
                    // Row index in C is relative to the start of C matrix.
                    let micro_panel_start_row_in_c = ic_offset_in_c + ir_idx * MR;
                    // Column index in C is relative to the start of c_jc_block_slice (which is jc_start of C).
                    let micro_panel_start_col_in_c_block = jr_idx * NR;

                    // Offset from the beginning of c_jc_block_slice to C(row_start, col_start_in_block)
                    let micropanel_start_offset_in_c_block_slice = at(
                        micro_panel_start_row_in_c,
                        micro_panel_start_col_in_c_block,
                        m_orig_c, // Use leading dimension of original C
                    );

                    // Calculate the end of the C micro-panel slice.
                    // This defines a dense region of `mr_eff_micropanel` rows and `nr_eff_micropanel` columns.
                    // NOTE: kernel_8x6 is hardcoded for 6 columns. If nr_eff_micropanel < 6,
                    // this slice might be too small for kernel_8x6 if it writes to all 6 columns.
                    // This implies kernel_8x6 must handle nr_eff_micropanel correctly for columns,
                    // or this slicing needs to account for the kernel's fixed width (e.g. use NR for width).
                    // Given "do not change code", documenting this behavior.
                    let slice_end_offset_exclusive = micropanel_start_offset_in_c_block_slice
                        + (nr_eff_micropanel.saturating_sub(1)) * m_orig_c // (cols-1)*ldc
                        + mr_eff_micropanel; // + rows

                    // Ensure the slice does not exceed the bounds of c_jc_block_slice.
                    let final_slice_end_exclusive =
                        slice_end_offset_exclusive.min(c_jc_block_slice.len());

                    // Get the mutable slice for the C micro-panel.
                    let c_micropanel_for_kernel = &mut c_jc_block_slice
                        [micropanel_start_offset_in_c_block_slice..final_slice_end_exclusive];

                    // Call the micro-kernel.
                    kernel_8x6(
                        a_panel_packed,
                        b_panel_packed,
                        c_micropanel_for_kernel,
                        mr_eff_micropanel,
                        nr_eff_micropanel,
                        kc_eff_common,
                        m_orig_c, // Pass leading dimension of C
                    );
                });
        });
}

/// Macro-kernel with fused B packing: C += A * B.
///
/// A is pre-packed. B is packed on-the-fly (panel by panel) into `block_b_packed_dest`.
/// This can improve cache performance by packing B just before it's used.
///
/// # Arguments
///
/// * `block_a_packed`:              Packed block of A.
/// * `b_block_original_data_slice`: Slice of original B data for the current block.
/// * `block_b_packed_dest`:         Mutable buffer to store packed B panels. Must be large enough
///   for one packed B block (`num_col_panels_in_b_block * kc_eff_common * NR`).
/// * `c_jc_block_slice`:            Mutable slice of C for the current JC block.
/// * `m_orig_c`:                    Leading dimension of original C matrix.
/// * `k_orig_b`:                    Leading dimension (rows) of original B matrix.
/// * `mc_eff_a_block`:              Effective rows in A block.
/// * `nc_eff_b_block`:              Effective columns in B block.
/// * `kc_eff_common`:               Effective common dimension (k).
/// * `ic_offset_in_c`:              Row offset in C for this A block's contribution.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macro_kernel_fused_b(
    block_a_packed: &[f32],
    b_block_original_data_slice: &[f32], // Source B data for the current (pc, jc) block
    block_b_packed_dest: &mut [f32],     // Destination for all packed B panels in this B block
    c_jc_block_slice: &mut [f32],
    m_orig_c: usize, // Leading dimension of C
    k_orig_b: usize, // Leading dimension of B (original k rows)
    mc_eff_a_block: usize,
    nc_eff_b_block: usize,
    kc_eff_common: usize,
    ic_offset_in_c: usize, // Starting row in C (ic_start)
) {
    const MR: usize = 8;
    const NR: usize = 6; // Must match kernel_8x6 and B packing strategy

    // Calculate the number of NR-wide column panels in the B block.
    let num_col_panels_in_b_block = nc_eff_b_block.msrv_div_ceil(NR);

    // Outer loop: Iterate over column panels of B.
    for jr_idx in 0..num_col_panels_in_b_block {
        // Effective number of columns for this specific B panel.
        let nr_eff_this_b_panel = min(NR, nc_eff_b_block - jr_idx * NR);

        if nr_eff_this_b_panel == 0 {
            continue;
        }

        // Determine the sub-slice within `block_b_packed_dest` for the current B panel.
        // Each B panel is (kc_eff_common rows) x (NR columns when packed).
        let b_panel_dest_offset_in_storage = jr_idx * (kc_eff_common * NR);
        let b_panel_packed_slice_mut = &mut block_b_packed_dest
            [b_panel_dest_offset_in_storage..b_panel_dest_offset_in_storage + kc_eff_common * NR];

        // Determine the source slice for the current B panel from the original B data.
        // `b_block_original_data_slice` points to B(pc_start, jc_start).
        // We need the sub-panel starting at column `jr_idx * NR` within this block.
        // The `pack_panel_b` function handles column-major source access.
        // The source offset is (jr_idx * NR) columns over, each column having k_orig_b stride.
        // However, pack_panel_b takes a slice that starts at B(pc_start, jc_start + jr_idx * NR).
        // `b_block_original_data_slice` is B(pc,jc). The panel starts at B(pc, jc + jr_idx*NR).
        // So `current_b_panel_source_offset` is for `at(0, jr_idx * NR, k_orig_b)` relative to `b_block_original_data_slice`.
        let current_b_panel_source_offset = at(0, jr_idx * NR, k_orig_b); // Offset relative to B(pc,jc)
        let current_b_panel_source_slice =
            &b_block_original_data_slice[current_b_panel_source_offset..];

        // Pack the current B panel into its designated spot in `b_panel_packed_slice_mut`.
        pack_panel_b(
            b_panel_packed_slice_mut,     // Dest for this one B panel
            current_b_panel_source_slice, // Source data for this B panel
            nr_eff_this_b_panel,
            kc_eff_common, // Number of rows in B panel (k dimension)
            k_orig_b,      // Leading dimension of original B matrix
        );

        // Inner loop: Iterate over row panels of A (already packed).
        block_a_packed
            .chunks(MR * kc_eff_common) // Each chunk is one packed A panel
            .enumerate()
            .for_each(|(ir_idx, a_panel_packed)| {
                // Effective number of rows for this A micro-panel.
                let mr_eff_micropanel = min(MR, mc_eff_a_block - ir_idx * MR);

                if mr_eff_micropanel == 0 {
                    return; // Skip if micro-panel is empty.
                }

                // Calculate start of the C micro-panel within c_jc_block_slice.
                let micro_panel_start_row_in_c = ic_offset_in_c + ir_idx * MR;
                let micro_panel_start_col_in_c_block = jr_idx * NR; // jr_idx is from the outer B panel loop

                let micropanel_start_offset_in_c_block_slice = at(
                    micro_panel_start_row_in_c,
                    micro_panel_start_col_in_c_block,
                    m_orig_c,
                );

                // Calculate the end of the C micro-panel slice.
                // nr_eff_this_b_panel is the nr_eff for the current micro_panel.
                // See NOTE in macro_kernel_standard about C slice sizing vs kernel's fixed width.
                let slice_end_offset_exclusive = micropanel_start_offset_in_c_block_slice
                    + (nr_eff_this_b_panel.saturating_sub(1)) * m_orig_c
                    + mr_eff_micropanel;

                let final_slice_end_exclusive =
                    slice_end_offset_exclusive.min(c_jc_block_slice.len());

                let c_micropanel_for_kernel = &mut c_jc_block_slice
                    [micropanel_start_offset_in_c_block_slice..final_slice_end_exclusive];

                // Call the micro-kernel.
                // `b_panel_packed_slice_mut` contains the just-packed B panel.
                kernel_8x6(
                    a_panel_packed,
                    b_panel_packed_slice_mut, // Use the freshly packed B panel
                    c_micropanel_for_kernel,
                    mr_eff_micropanel,
                    nr_eff_this_b_panel, // nr_eff for this specific micro-operation
                    kc_eff_common,
                    m_orig_c,
                );
            });
    }
}

// --- Main Matmul Function ---

/// Performs matrix multiplication C = A * B using a block-based approach with packing.
///
/// Matrices A, B, and C are assumed to be in column-major order.
/// The multiplication is broken down into blocks of size MCxKC (for A) and KCxNC (for B).
/// Panels of A and B are packed into cache-friendly formats before being processed by
/// a SIMD micro-kernel.
///
/// This implementation uses a strategy where:
/// - A is packed per (IC, PC) block.
/// - B is packed per (PC, JC) block. The first time a B block is needed for a given (PC, JC),
///   it's packed using `macro_kernel_fused_b`. For subsequent uses within the same (PC, JC)
///   (i.e., different IC blocks), the already packed B is reused via `macro_kernel_standard`.
///
/// # Arguments
///
/// * `a`: Slice containing matrix A (column-major, M rows, K columns).
/// * `b`: Slice containing matrix B (column-major, K rows, N columns).
/// * `c`: Mutable slice for matrix C (column-major, M rows, N columns). C is updated: C += A*B.
///   It's assumed C is initialized (e.g., to zeros if a pure A*B is desired).
/// * `m`: Number of rows in A and C.
/// * `n`: Number of columns in B and C.
/// * `k`: Number of columns in A and rows in B (common dimension).
///
/// # Safety
///
/// This function is marked with `#[target_feature(enable = "avx2,avx,fma")]`.
/// The caller must ensure that the CPU supports these features at runtime.
/// Failure to do so when this function is compiled with these features enabled
/// could lead to illegal instruction errors.
/// The `unsafe` blocks within helper functions (e.g., pointer operations, SIMD intrinsics)
/// rely on correct slice indexing and data layouts.
#[target_feature(enable = "avx2,avx,fma")] // Enable AVX2, AVX, and FMA for this function
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 {
        // If M or N is zero, the result matrix is empty or operations are trivial.
        // K can be zero, in which case A*B is a zero matrix (if C is init to 0).
        return;
    }

    // Loop over C by NC-sized column blocks (jc_start)
    for jc_start in (0..n).step_by(NC) {
        let nc_eff_block = min(NC, n - jc_start); // Effective width of B and C block

        // Loop over A/B by KC-sized K-dimension blocks (pc_start)
        for pc_start in (0..k).step_by(KC) {
            let kc_eff_block = min(KC, k - pc_start); // Effective depth of A and B block

            if kc_eff_block == 0 {
                // If k-dimension block is empty, this step contributes nothing.
                // This can happen if k is not a multiple of KC and we are at the last, smaller k-block.
                // Or if k itself is 0.
                continue;
            }

            // --- B Block Handling ---
            // Source slice for the current B block B(pc_start:pc_start+kc_eff_block-1, jc_start:jc_start+nc_eff_block-1)
            let b_block_original_data_offset = at(pc_start, jc_start, k); // k is ldb (rows in B)
                                                                          // Ensure slice does not go out of bounds for B
            let b_block_end_offset = at(pc_start, jc_start + nc_eff_block, k) // Start of col after block
                .min(b.len()); // Cap at B's total length
            let b_block_original_data_slice = &b[b_block_original_data_offset..b_block_end_offset];

            // Storage for the packed B block. This B block will be reused for multiple A blocks.
            let num_col_panels_in_b_block = nc_eff_block.msrv_div_ceil(NR);
            let packed_b_storage_size = num_col_panels_in_b_block * kc_eff_block * NR;
            let mut packed_b_storage =
                alloc_zeroed_f32_vec(packed_b_storage_size, f32x8::AVX_ALIGNMENT);

            // Flag to track if the current B block (for this jc_start, pc_start) has been packed.
            let mut b_block_has_been_packed_this_jc_pc = false;

            // Loop over C/A by MC-sized row blocks (ic_start)
            for ic_start in (0..m).step_by(MC) {
                let mc_eff_block = min(MC, m - ic_start); // Effective height of A and C block

                if mc_eff_block == 0 {
                    continue; // If m-dimension block is empty.
                }

                // --- A Block Handling ---
                // Source slice for the current A block A(ic_start:ic_start+mc_eff_block-1, pc_start:pc_start+kc_eff_block-1)
                let a_block_original_data_offset = at(ic_start, pc_start, m); // m is lda (rows in A)
                                                                              // Ensure slice does not go out of bounds for A
                let a_block_end_offset = at(ic_start, pc_start + kc_eff_block, m) // Start of col after block
                    .min(a.len()); // Cap at A's total length
                let a_block_original_data_slice =
                    &a[a_block_original_data_offset..a_block_end_offset];

                // Pack the A block. This is done for every A block.
                let block_a_packed = pack_block_a(
                    a_block_original_data_slice,
                    mc_eff_block,
                    kc_eff_block,
                    m, // LDA (rows of original A)
                );

                // --- C Block Handling ---
                // Target slice in C for the current column block defined by jc_start and nc_eff_block.
                // This slice covers all M rows for these nc_eff_block columns.
                // C(0:m-1, jc_start:jc_start+nc_eff_block-1)
                let c_jc_block_slice_start = at(0, jc_start, m); // m is ldc (rows in C)
                                                                 // The slice for C columns from jc_start to jc_start + nc_eff_block - 1, all m rows.
                                                                 // Total elements = m * nc_eff_block.
                let c_jc_block_slice_end = (c_jc_block_slice_start + m * nc_eff_block).min(c.len());
                let c_target_jc_block_slice = &mut c[c_jc_block_slice_start..c_jc_block_slice_end];

                // --- Macro Kernel Call ---
                if !b_block_has_been_packed_this_jc_pc {
                    // First time encountering this B block (for fixed jc_start, pc_start)
                    // Use fused kernel: packs B on-the-fly into `packed_b_storage`.
                    macro_kernel_fused_b(
                        &block_a_packed,
                        b_block_original_data_slice, // Unpacked B data for this block
                        &mut packed_b_storage,       // Destination for packed B
                        c_target_jc_block_slice,     // C block to update
                        m,                           // LDC (original rows in C)
                        k,                           // LDB (original rows in B)
                        mc_eff_block,
                        nc_eff_block,
                        kc_eff_block,
                        ic_start, // Row offset in C for this A block's output
                    );
                    // Mark B as packed if the operation was non-trivial.
                    if mc_eff_block > 0 && kc_eff_block > 0 && nc_eff_block > 0 {
                        // Ensure B was actually needed and packed
                        b_block_has_been_packed_this_jc_pc = true;
                    }
                } else {
                    // B block is already packed from a previous ic_start iteration. Reuse it.
                    macro_kernel_standard(
                        &block_a_packed,
                        &packed_b_storage, // Use already packed B data
                        c_target_jc_block_slice,
                        m, // LDC
                        mc_eff_block,
                        nc_eff_block,
                        kc_eff_block,
                        ic_start,
                    );
                }
            } // End loop ic_start (MC)
        } // End loop pc_start (KC)
    } // End loop jc_start (NC)
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
