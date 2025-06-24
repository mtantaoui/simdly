use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::cmp::min;

use crate::{
    simd::{
        avx512::f32x16::{self, F32x16, AVX512_ALIGNMENT},
        traits::SimdVec,
        utils::alloc_zeroed_f32_vec,
        // utils::alloc_uninit_f32_vec,
    },
    KC, MC, MR, NC, NR,
};

/// Packs a panel of matrix B directly into a provided destination slice.
/// Matrix B is assumed to be in column-major format. The `NR` constant,
/// typically representing a SIMD register width or micro-kernel dimension,
/// dictates the width of the packed rows in the destination.
///
/// ## Packing Logic and Output Layout
///
/// This function processes the input `b_panel_source_slice` (which represents
/// a `kc_panel` x `nr_effective_in_panel` panel from the original matrix B)
/// and packs it into `dest_slice`. The packing is "row by row" from the
/// perspective of this panel.
///
/// Specifically, for each "row" `p_row_in_panel` (from `0` to `kc_panel - 1`)
/// of the input panel:
/// 1. It copies `nr_effective_in_panel` elements, corresponding to
///    `B_panel(p_row_in_panel, 0)` through `B_panel(p_row_in_panel, nr_effective_in_panel - 1)`,
///    into `dest_slice`.
/// 2. These `nr_effective_in_panel` elements are followed by `NR - nr_effective_in_panel`
///    zeros to pad this packed segment to a total length of `NR`.
///
/// These `kc_panel` packed segments (each of length `NR`) are laid out contiguously
/// in `dest_slice`.
///
/// The memory layout within `dest_slice` (relative to its own start) will be:
/// ```
/// // p_row_in_panel = 0:
/// // [ B_panel(0,0), B_panel(0,1), ..., B_panel(0,nr_eff-1), 0.0, ..., 0.0 ] (NR elements total)
/// //
/// // p_row_in_panel = 1:
/// // [ B_panel(1,0), B_panel(1,1), ..., B_panel(1,nr_eff-1), 0.0, ..., 0.0 ] (NR elements total)
/// //
/// // ...
/// //
/// // p_row_in_panel = kc_panel - 1:
/// // [ B_panel(kc-1,0), ..., B_panel(kc-1,nr_eff-1), 0.0, ..., 0.0 ] (NR elements total)
/// ```
/// where `B_panel(row, col)` refers to an element from the conceptual `kc_panel` x `nr_effective_in_panel`
/// input panel that `b_panel_source_slice` represents.
///
/// This layout is effectively a row-major storage of the `kc_panel` x `nr_effective_in_panel` panel,
/// where each "row" of the panel is padded to `NR` elements.
///
/// # Arguments
///
/// * `dest_slice` - A mutable slice where the packed panel will be written.
///   Its length **MUST** be `kc_panel * NR`. The `NR` value is typically a `const`
///   defined in the surrounding module.
/// * `b_panel` - A slice into the original column-major matrix B's data.
///   This slice must start at the memory location corresponding to the top-left
///   element of the conceptual `kc_panel` x `nr_effective_in_panel` panel being packed.
///   For example, if packing `B(pc, jc)` to `B(pc+kc_panel-1, jc+nr_effective_in_panel-1)`,
///   this slice should start at the memory address of `B(pc, jc)`.
/// * `nr` - The actual number of columns to pack from the source panel.
///   Must satisfy `nr_effective_in_panel <= NR`.
/// * `kc` - The number of "rows" (or depth along the K-dimension) to pack
///   from the source panel. This dictates how many `NR`-element segments will be written.
/// * `k` - The leading dimension (total number of rows) of the
///   original, full matrix B. This is crucial for correctly calculating offsets
///   between columns in the column-major `b_panel_source_slice`.
///
/// # Panics
///
/// * If `dest_slice.len()` is not equal to `kc_panel * NR`.
/// * If `nr_effective_in_panel > NR`.
/// * If `NR == 0` (this is usually checked via an assert on the `const NR` elsewhere).
/// * If `b_panel_source_slice` is too short for the read accesses implied by
///   `nr_effective_in_panel`, `kc_panel`, and `k_original_matrix`, leading to an
///   out-of-bounds panic during `b_panel_source_slice[source_index]`.
#[inline(always)]
fn pack_panel_b(
    // Consider renaming to pack_panel_b_into if it's an internal "into" variant
    dest_slice: &mut [f32],
    b_panel: &[f32],
    nr: usize,
    kc: usize,
    k: usize,
) {
    // NR is assumed to be a const in scope, e.g., const NR: usize = 4;
    debug_assert_eq!(
        dest_slice.len(),
        kc * NR,
        "Destination slice length incorrect. Expected {}, got {}. kc_panel={}, NR={}",
        kc * NR,
        dest_slice.len(),
        kc,
        NR // Assuming NR is a const usize in scope
    );
    debug_assert!(
        nr <= NR, // Assuming NR is a const usize in scope
        "nr_effective_in_panel ({nr}) cannot exceed NR ({NR})" // Assuming NR is a const usize in scope
    );
    // Implicitly, NR > 0 is expected for the padding loop `nr_effective_in_panel..NR` to make sense.

    // Iterate over the `kc_panel` "rows" of the panel. Each `p_row_in_panel`
    // corresponds to a specific K-index offset (e.g., pc + p_row_in_panel)
    // in the original matrix B.
    for p_row_in_panel in 0..kc {
        // Calculate the starting index in `dest_slice` for the current "row" `p_row_in_panel`.
        // Each packed "row" occupies `NR` elements.
        let dest_row_start_index = p_row_in_panel * NR;

        // Iterate over the `nr_effective_in_panel` actual columns of the panel B that we need to copy.
        // `j_col_in_panel` here corresponds to the column index within the sub-panel
        // (e.g., jc + j_col_in_panel in original matrix).
        for j_col_in_panel in 0..nr {
            // Accessing B_panel(p_row_in_panel, j_col_in_panel) from `b_panel_source_slice`.
            // The input slice `b_panel_source_slice` starts at B_panel(0,0) of this panel.
            // In a column-major matrix with leading dimension `k_original_matrix`, an element at
            // panel-relative (row `p_row_in_panel`, col `j_col_in_panel`) is located at
            // memory offset `j_col_in_panel * k_original_matrix + p_row_in_panel`
            // within `b_panel_source_slice`.
            let source_index = j_col_in_panel * k + p_row_in_panel;

            // Copy the element from the source panel to the destination packed slice.
            dest_slice[dest_row_start_index + j_col_in_panel] = b_panel[source_index];
        }

        // Pad the remainder of the current `NR`-width segment in `dest_slice` with zeros.
        // This loop runs if `nr_effective_in_panel < NR`.
        for j_pad_col in nr..NR {
            dest_slice[dest_row_start_index + j_pad_col] = 0.0;
        }
    }
}

/// Packs a block of matrix B by iteratively packing NR-column-wide panels.
/// Matrix B is assumed to be in column-major format., optimized to use an internal helper that writes directly
/// into slices of a single pre-allocated vector for the entire packed block,
/// minimizing allocations and copies.
///
/// # Arguments
///
/// * `b` - A slice representing a block of matrix B, `B(pc..pc+kc-1, jc..jc+nc-1)`.
///   This slice must start at the memory location corresponding to `B(pc, jc)`.
/// * `nc` - The total number of columns in this block of B (width of the block).
/// * `kc` - The total number of rows (depth) in this block of B (height of the block).
/// * `k` - The leading dimension (number of rows) of the original matrix B.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed block of B. The total size will be
/// `ceil(nc / NR) * kc * NR`.
///
/// # Panics
/// * If `NR == 0` (this is a `const` so it's a compile-time consideration).
/// * If internal slicing or preconditions for `pack_one_panel_into_dest` are violated,
///   typically due to `b` being too short for the given `nc`, `kc`, `k` dimensions.
#[inline(always)]
fn pack_block_b(b: &[f32], nc: usize, kc: usize, k: usize) -> Vec<f32> {
    // Handle edge cases: if the block has no rows or no columns, return an empty packed vector.
    if kc == 0 || nc == 0 {
        return Vec::new();
    }

    // Calculate the total number of NR-wide panels needed to cover `nc_block` columns.
    let num_panels_across_nc = nc.div_ceil(NR); // ceil(nc_block / NR)

    // Calculate the total size needed for the packed block.
    // Each of the `num_panels_across_nc` panels will result in `kc_block * NR` packed elements.
    let total_packed_size = num_panels_across_nc * kc * NR;

    // Allocate the vector for the entire packed block. Initialize with zeros.
    // `pack_one_panel_into_dest` will fill in data and re-fill padding with zeros.
    // let mut packed_block_b_data = vec![0.0f32; total_packed_size];
    let mut packed_block_b_data = alloc_zeroed_f32_vec(total_packed_size, f32x16::AVX512_ALIGNMENT);

    // `current_write_offset` tracks where the next packed panel should start in `packed_block_b_data`.
    let mut current_write_offset = 0;

    // Iterate over the columns of the input block `b_block_source_slice` in steps of `NR`.
    // `j_col_block_start` is the starting column index *within the current block*
    // for the panel being processed.
    for j_col_block_start in (0..nc).step_by(NR) {
        // Determine the number of effective columns for the current panel.
        let nr_effective_in_panel = min(NR, nc - j_col_block_start);

        // Determine the slice of `b_block_source_slice` that corresponds to the current panel.
        // The panel starts at column `j_col_block_start` within the block.
        // In the column-major `b_block_source_slice` (which starts at B(pc,jc)),
        // this is at memory offset `j_col_block_start * k_original_matrix`.
        let panel_source_data_start_index = j_col_block_start * k;
        let current_panel_source_slice = &b[panel_source_data_start_index..];

        // Determine the destination slice within `packed_block_b_data` for the current panel.
        let panel_dest_slice_end = current_write_offset + kc * NR;
        let dest_sub_slice = &mut packed_block_b_data[current_write_offset..panel_dest_slice_end];

        // Pack the current panel directly into its designated part of `packed_block_b_data`.
        // The internal helper `pack_one_panel_into_dest` uses the const NR.
        pack_panel_b(
            dest_sub_slice,
            current_panel_source_slice,
            nr_effective_in_panel,
            kc,
            k,
        );

        // Advance the write offset for the next panel.
        current_write_offset = panel_dest_slice_end;
    }

    packed_block_b_data
}

/// Packs a panel of matrix A directly into a provided destination slice.
/// Matrix A is assumed to be in column-major format.
/// The destination slice `dest_slice` is expected to be `kc_panel * MR` in length.
///
/// ## Packed Format (within dest_slice)
/// For each `p` from `0..kc_panel-1`:
///   `dest_slice[p*MR .. p*MR + mr_effective_in_panel-1]` gets `A_panel(0..mr_effective_in_panel-1, p)`
///   `dest_slice[p*MR + mr_effective_in_panel .. (p+1)*MR-1]` gets zeros (padding).
///
/// # Arguments
///
/// * `dest_slice` - Mutable slice where the packed panel is written. Length must be `kc_panel * MR`.
/// * `a_panel_source_slice` - Slice into original column-major A, starting at `A(ic_panel, pc_panel)`.
/// * `mr_effective_in_panel` - Actual number of rows to pack from this panel (`<= MR`).
/// * `kc_panel` - Number of columns to pack from this panel (depth).
/// * `m_original_matrix` - Leading dimension (rows) of the original matrix A.
#[inline(always)]
fn pack_panel_a_into(
    dest_slice: &mut [f32],
    a_panel_source_slice: &[f32],
    mr_effective_in_panel: usize,
    kc_panel: usize,
    m_original_matrix: usize,
) {
    debug_assert_eq!(
        dest_slice.len(),
        kc_panel * MR,
        "Destination slice length mismatch."
    );
    debug_assert!(
        mr_effective_in_panel <= MR,
        "mr_effective_in_panel cannot exceed MR."
    );

    for p_col_in_panel in 0..kc_panel {
        // Iterate over columns of the panel (K-dimension)
        let dest_col_segment_start = p_col_in_panel * MR;
        for i_row_in_panel in 0..mr_effective_in_panel {
            // Iterate over rows to copy for this column
            // Access A(ic_panel + i_row_in_panel, pc_panel + p_col_in_panel)
            // from a_panel_source_slice which starts at A(ic_panel, pc_panel)
            let source_index = p_col_in_panel * m_original_matrix + i_row_in_panel;
            dest_slice[dest_col_segment_start + i_row_in_panel] =
                a_panel_source_slice[source_index];
        }
        // Pad with zeros if mr_effective_in_panel < MR
        for i_pad in mr_effective_in_panel..MR {
            dest_slice[dest_col_segment_start + i_pad] = 0.0;
        }
    }
}

/// Packs a block of matrix A by iteratively packing MR-row-high panels.
/// Matrix A is assumed to be in column-major format.
/// This version is optimized to use `pack_panel_a_into` for direct writing.
///
/// # Arguments
///
/// * `a` - Slice representing `A(ic_block..ic_block+mc_block-1, pc_block..pc_block+kc_block-1)`.
///   Starts at memory location of `A(ic_block, pc_block)`.
/// * `mc` - Total number of rows in this block of A.
/// * `kc` - Total number of columns (depth) in this block of A.
/// * `m` - Leading dimension (rows) of the original matrix A.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed block of A. Size is `ceil(mc_block / MR) * kc_block * MR`.
#[inline(always)]
fn pack_block_a(a: &[f32], mc: usize, kc: usize, m: usize) -> Vec<f32> {
    if mc == 0 || kc == 0 {
        return Vec::new();
    }

    let num_row_panels = mc.div_ceil(MR); // ceil(mc_block / MR)
    let total_packed_size = num_row_panels * kc * MR;

    // let mut packed_block_a_data = vec![0.0f32; total_packed_size];

    let mut packed_block_a_data = alloc_zeroed_f32_vec(total_packed_size, AVX512_ALIGNMENT);

    let mut current_write_offset = 0;

    for i_row_block_start in (0..mc).step_by(MR) {
        // Iterate over rows of A_block in MR-sized steps
        let mr_effective = min(MR, mc - i_row_block_start);

        // The slice for pack_panel_a_into should start at A(ic_block + i_row_block_start, pc_block).
        // Since a_block_source_slice starts at A(ic_block, pc_block),
        // the offset is `i_row_block_start` (for column-major access to the first element of that row).
        let panel_source_slice = &a[i_row_block_start..];

        let dest_slice_len_for_panel = kc * MR;
        let dest_sub_slice = &mut packed_block_a_data
            [current_write_offset..current_write_offset + dest_slice_len_for_panel];

        pack_panel_a_into(
            dest_sub_slice,
            panel_source_slice,
            mr_effective,
            kc, // kc_block is the number of columns for the panel
            m,
        );
        current_write_offset += dest_slice_len_for_panel;
    }

    packed_block_a_data
}

/// Performs matrix multiplication `C = A * B` using a blocked algorithm.
/// Matrices A, B, and C are assumed to be in column-major format.
///
/// The algorithm follows the GotoBLAS/BLIS approach:
/// 1. Loop `jc` over columns of C (and B) in `NC` steps. (Outer loop)
/// 2. Loop `pc` over the K dimension (cols of A, rows of B) in `KC` steps. (Packing loop)
/// 3. Loop `ic` over rows of C (and A) in `MC` steps. (Inner loop)
///
/// Inside these loops:
///   - A block of B (`nc x kc`) is packed (`block_b_packed`).
///   - For each `ic` step:
///     - A block of A (`mc x kc`) is packed (`block_a_packed`).
///     - Micro-kernel operations:
///       - Loop `jr` over columns of `block_b_packed` in `NR` steps.
///       - Loop `ir` over rows of `block_a_packed` in `MR` steps.
///       - Call `kernel_8x1` to compute an `mr x nr` sub-micro-panel of C.
///
/// # Arguments
///
/// * `a` - Slice containing matrix A (M x K, column-major).
/// * `b` - Slice containing matrix B (K x N, column-major).
/// * `c` - Mutable slice for matrix C (M x N, column-major), to store the result.
/// * `m` - Number of rows of A and C.
/// * `n` - Number of columns of B and C.
/// * `k` - Number of columns of A and rows of B.
///
/// # Safety
///
/// .
#[target_feature(enable = "avx512f,avx512vl")]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Loop over columns of C by NC blocks
    // `c_chunk` is a mutable slice of C representing `m` rows and `nc` columns, starting at column `jc`.
    // It's equivalent to C(0..m-1, jc..jc+nc-1).
    // In column-major, this is `m * nc` contiguous elements if C were just this chunk.
    // Here, `c.chunks_mut(m * NC)` splits C into major column blocks. Each chunk is `m * NC` elements.
    // However, C is a flat slice representing the whole matrix. If m < NC, m * NC is not the right chunk size.
    // The chunk size should be for one block of columns in C, which is m rows * NC cols.
    // If C has N columns total, and we iterate jc by NC:
    // For jc = 0, we process C(:, 0..NC-1). This is m * NC elements.
    // For jc = NC, we process C(:, NC..2*NC-1). This is m * NC elements.
    // So c.chunks_mut(m * NC) is correct.
    c.chunks_mut(m * NC)
        .enumerate()
        .for_each(|(j_idx, c_chunk)| {
            let jc = j_idx * NC; // Current column offset in C and B
            let nc = min(NC, n - jc); // Effective width of the current C_block and B_block

            // Loop over the K dimension by KC blocks (packing loop)
            // `a_chunk_for_k_strip` corresponds to A(0..m-1, pc..pc+kc-1)
            // In column-major, this is `m * kc` contiguous elements if A were just this chunk.
            // `a.chunks(m * KC)` splits A into major column blocks.
            // This assumes A is laid out contiguously column-block by column-block.
            // If A is M x K, a chunk is A(:, pc..pc+KC-1). This has m rows and KC columns. Size m * KC.
            // This is correct.
            a.chunks(m * KC) // Each chunk is A(:, pc*KC .. (pc+1)*KC-1 )
                .enumerate()
                .for_each(|(p_idx, a_col_strip_for_k)| {
                    // Renamed for clarity
                    let pc = p_idx * KC; // Current K offset
                    let kc = min(KC, k - pc); // Effective depth of current A_block and B_block

                    // Pack a block of B: B(pc..pc+kc-1, jc..jc+nc-1)
                    // Offset into B is `jc * k + pc`: `jc` columns over (each `k` elements long), then `pc` rows down.
                    // This correctly points to B(pc, jc).
                    let b_block_slice = &b[at(pc, jc, k)..];
                    let block_b_packed = pack_block_b(b_block_slice, nc, kc, k);

                    // Loop over rows of C by MC blocks (innermost loop of the three macro-loops)
                    for ic in (0..m).step_by(MC) {
                        let mc = min(MC, m - ic); // Effective height of current C_block and A_block

                        let block_a_packed = pack_block_a(&a_col_strip_for_k[ic..], mc, kc, m);

                        // Micro-kernel operations iterate over the packed blocks
                        // Loop over sub-columns of B_block (packed) by NR steps
                        block_b_packed
                            .chunks(kc * NR) // Each chunk is a panel of B, `kc * NR` elements
                            .enumerate()
                            .for_each(|(jr_idx, b_panel)| {
                                // jr_idx is micro-panel column index
                                // Loop over sub-rows of A_block (packed) by MR steps
                                block_a_packed
                                    .chunks(MR * kc) // Each chunk is a panel of A, `MR * kc` elements
                                    .enumerate()
                                    .for_each(|(ir_idx, a_panel)| {
                                        // ir_idx is micro-panel row index
                                        let nr_eff = min(NR, nc - jr_idx * NR); // Effective num cols for this micro-op
                                        let mr_eff = min(MR, mc - ir_idx * MR); // Effective num rows for this micro-op

                                        // Calculate pointer to the C micropanel C(ic+ir*MR, jc+jr*NR)
                                        // This micropanel is mr_eff x nr_eff.
                                        // c_chunk points to C(0..m-1, jc..jc+nc-1).
                                        // Offset: (jr_idx * NR) columns over, each `m` elements long.
                                        // Then (ic + ir_idx * MR) rows down.
                                        // Global row: ic_abs = ic + ir_idx * MR
                                        // Global col: jc_abs = jc + jr_idx * NR
                                        // Index in C: at(ic_abs, jc_abs, m)
                                        // Index in c_chunk: C(ic_abs, (jc_abs - jc), m_of_c_chunk = m)
                                        // jr_chunk_offset = jr_idx * NR
                                        // ir_chunk_offset = ir_idx * MR
                                        // c_micropanel_start_idx in c_chunk for C(ic + ir_chunk_offset, jr_chunk_offset)
                                        let c_micropanel_start_idx_in_c_chunk =
                                            at(ic + ir_idx * MR, jr_idx * NR, m);

                                        let c_micropanel =
                                            &mut c_chunk[c_micropanel_start_idx_in_c_chunk..];

                                        // Perform the core computation C_micropanel += A_panel * B_panel
                                        kernel_16x4(
                                            a_panel,
                                            b_panel,
                                            c_micropanel,
                                            mr_eff,
                                            nr_eff,
                                            kc,
                                            m,
                                        );
                                    });
                            });
                    }
                });
        });
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

/// Performs matrix multiplication `C = A * B` in parallel using Rayon.
/// Matrices A, B, and C are assumed to be in column-major format.
///
/// Parallelism strategy:
/// - The outermost loop over `jc` (columns of C, `NC` blocks) is parallelized.
///   Each thread gets an independent `c_chunk`.
/// - The loop over `pc` (K-dimension, `KC` blocks) is serial within each `jc` task.
/// - The loop over `jr` (micro-columns of B_block, `NR` steps) is serial within each `jc` task.
///
/// The doc comment in the original problem description suggested more parallelism levels
/// which would require synchronization (like the commented-out Mutex). The current code,
/// however, only parallelizes the `jc` loop, making `c_chunk` exclusive to each task.
///
/// # Arguments
///
/// * `a` - Slice containing matrix A (M x K, column-major).
/// * `b` - Slice containing matrix B (K x N, column-major).
/// * `c` - Mutable slice for matrix C (M x N, column-major), to store the result.
/// * `m` - Number of rows of A and C.
/// * `n` - Number of columns of B and C.
/// * `k` - Number of columns of A and rows of B.
///
/// # Safety
///
/// .
#[target_feature(enable = "avx512f,avx512vl")]
pub fn par_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.par_chunks_mut(m * NC)
        .enumerate()
        .for_each(|(j_idx, c_chunk)| {
            let jc = j_idx * NC;
            let nc = min(NC, n - jc);

            // Inner loops are serial for each c_chunk task. No Mutex needed for c_chunk here.
            a.chunks(m * KC) // Standard iterator
                .enumerate()
                .for_each(|(p_idx, a_col_strip_for_k)| {
                    let pc = p_idx * KC;
                    let kc = min(KC, k - pc);

                    let b_block_slice = &b[at(pc, jc, k)..];
                    let block_b_packed = pack_block_b(b_block_slice, nc, kc, k);

                    for ic in (0..m).step_by(MC) {
                        let mc = min(MC, m - ic);

                        let block_a_packed = pack_block_a(&a_col_strip_for_k[ic..], mc, kc, m);

                        block_b_packed
                            .chunks(kc * NR) // Standard iterator
                            .enumerate()
                            .for_each(|(jr_idx, b_panel)| {
                                block_a_packed
                                    .chunks(MR * kc) // Standard iterator
                                    .enumerate()
                                    .for_each(|(ir_idx, a_panel)| {
                                        let nr_eff = min(NR, nc - jr_idx * NR);
                                        let mr_eff = min(MR, mc - ir_idx * MR);

                                        let c_micropanel_start_idx_in_c_chunk =
                                            at(ic + ir_idx * MR, jr_idx * NR, m);
                                        let c_micropanel =
                                            &mut c_chunk[c_micropanel_start_idx_in_c_chunk..];

                                        kernel_16x4(
                                            a_panel,
                                            b_panel,
                                            c_micropanel,
                                            mr_eff,
                                            nr_eff,
                                            kc,
                                            m,
                                        );
                                    });
                            });
                    }
                });
        });
}

#[allow(dead_code, clippy::needless_range_loop, non_snake_case)]
fn kernel_MRxNR(
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro_panel: &mut [f32],
    mr_eff: usize,
    nr_eff: usize,
    kc: usize,
    ldc: usize,
) {
    debug_assert_eq!(MR, 16);
    debug_assert!(
        mr_eff <= MR,
        "mr_eff ({mr_eff}) must be less than or equal to MR ({MR})"
    );
    debug_assert!(
        nr_eff <= NR,
        "nr_eff ({nr_eff}) must be less than or equal to NR ({NR})"
    );

    let mut c_simd_cols: [F32x16; NR] = [unsafe { F32x16::splat(0.0) }; NR]; // Max NR cols

    // Load C columns into SIMD registers
    for j_col in 0..nr_eff {
        let c_col_ptr = unsafe { c_micro_panel.as_ptr().add(j_col * ldc) };
        c_simd_cols[j_col] = match mr_eff.cmp(&MR) {
            std::cmp::Ordering::Equal => unsafe { F32x16::load(c_col_ptr, mr_eff) },
            std::cmp::Ordering::Less => unsafe { F32x16::load_partial(c_col_ptr, mr_eff) },
            std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
        };
    }

    for p in 0..kc {
        // Loop over K dimension
        let a_simd_vec = F32x16::new(&a_panel[p * MR..p * MR + MR]);

        for j_col in 0..nr_eff {
            // Loop over N dimension (columns of B and C micro-panel)
            let b_scalar = b_panel[p * NR + j_col]; // Access correct B element for current column
            let b_simd_splat = unsafe { F32x16::splat(b_scalar) };
            c_simd_cols[j_col] = unsafe { c_simd_cols[j_col].fmadd(a_simd_vec, b_simd_splat) };
        }
    }

    // Store C columns back
    for j_col in 0..nr_eff {
        let c_col_ptr = unsafe { c_micro_panel.as_mut_ptr().add(j_col * ldc) };
        match mr_eff.cmp(&MR) {
            std::cmp::Ordering::Equal => unsafe { c_simd_cols[j_col].store_at(c_col_ptr) },
            std::cmp::Ordering::Less => unsafe { c_simd_cols[j_col].store_at_partial(c_col_ptr) },
            std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
        };
    }
}

#[inline(always)]
fn kernel_16x4(
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro_panel: &mut [f32],
    mr_eff: usize,
    nr_eff: usize,
    kc: usize,
    ldc: usize,
) {
    debug_assert_eq!(MR, 16);
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
    // let c_col5_ptr = unsafe { c_micro_panel.as_mut_ptr().add(4 * ldc) };
    // let c_col6_ptr = unsafe { c_micro_panel.as_mut_ptr().add(5 * ldc) };

    let (
        mut c_col1,
        mut c_col2,
        mut c_col3,
        mut c_col4,
        //  mut c_col5, mut c_col6
    ) = match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => unsafe {
            (
                F32x16::load(c_col1_ptr, mr_eff),
                F32x16::load(c_col2_ptr, mr_eff),
                F32x16::load(c_col3_ptr, mr_eff),
                F32x16::load(c_col4_ptr, mr_eff),
                // F32x16::load(c_col5_ptr, mr_eff),
                // F32x16::load(c_col6_ptr, mr_eff),
            )
        },
        std::cmp::Ordering::Less => unsafe {
            (
                F32x16::load_partial(c_col1_ptr, mr_eff),
                F32x16::load_partial(c_col2_ptr, mr_eff),
                F32x16::load_partial(c_col3_ptr, mr_eff),
                F32x16::load_partial(c_col4_ptr, mr_eff),
                // F32x16::load_partial(c_col5_ptr, mr_eff),
                // F32x16::load_partial(c_col6_ptr, mr_eff),
            )
        },
        std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
    };

    let mut b_scalar_splat: F32x16;

    for p in 0..kc {
        // Loop over K dimension
        let a_col = F32x16::new(&a_panel[p * MR..p * MR + MR]);

        // Loop over N dimension (columns of B and C micro-panel)

        b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR]) };
        c_col1 = unsafe { c_col1.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR + 1]) };
        c_col2 = unsafe { c_col2.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR + 2]) };
        c_col3 = unsafe { c_col3.fmadd(a_col, b_scalar_splat) };

        b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR + 3]) };
        c_col4 = unsafe { c_col4.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR + 4]) };
        // c_col5 = unsafe { c_col5.fmadd(a_col, b_scalar_splat) };

        // b_scalar_splat = unsafe { F32x16::splat(b_panel[p * NR + 5]) };
        // c_col6 = unsafe { c_col6.fmadd(a_col, b_scalar_splat) };
    }

    // Store C columns back
    match mr_eff.cmp(&MR) {
        std::cmp::Ordering::Equal => unsafe {
            c_col1.store_at(c_col1_ptr);
            c_col2.store_at(c_col2_ptr);
            c_col3.store_at(c_col3_ptr);
            c_col4.store_at(c_col4_ptr);
            // c_col5.store_at(c_col5_ptr);
            // c_col6.store_at(c_col6_ptr);
        },
        std::cmp::Ordering::Less => unsafe {
            c_col1.store_at_partial(c_col1_ptr);
            c_col2.store_at_partial(c_col2_ptr);
            c_col3.store_at_partial(c_col3_ptr);
            c_col4.store_at_partial(c_col4_ptr);
            // c_col5.store_at_partial(c_col5_ptr);
            // c_col6.store_at_partial(c_col6_ptr);
        },
        std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
    };
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
        let mut c_data = alloc_zeroed_f32_vec(m * n, f32x16::AVX512_ALIGNMENT);

        let expected_c = naive_matmul(&a_data, &b_data, m, n, k);

        if use_par {
            unsafe { par_matmul(&a_data, &b_data, &mut c_data, m, n, k) };
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
        let mut c_data = alloc_zeroed_f32_vec(m * n, f32x16::AVX512_ALIGNMENT);

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
