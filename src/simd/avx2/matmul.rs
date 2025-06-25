use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
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
        (self + rhs - 1) / rhs
    }
}

// --- Panel Packing Functions ---

/// Packs a panel of matrix A (column-major) into dest_slice.
/// Relies on dest_slice being pre-zeroed for padding.
#[inline(always)]
fn pack_panel_a_into(
    dest_slice: &mut [f32],       // Pre-zeroed, length kc_panel * MR
    a_panel_source_slice: &[f32], // Points to A(ic_panel_start, pc_panel_start for this panel)
    mr_effective_in_panel: usize, // <= MR
    kc_panel: usize,              // Number of columns in A's panel to pack (K-dim)
    m_original_matrix: usize,     // Leading dimension of original matrix A
) {
    debug_assert_eq!(dest_slice.len(), kc_panel * MR, "Dest A slice len mismatch");
    debug_assert!(mr_effective_in_panel <= MR, "mr_eff_a > MR");

    for p_col_in_panel in 0..kc_panel {
        // Iterate over columns of A's panel (K-dimension)
        // Offset in a_panel_source_slice to the start of the current source column.
        // a_panel_source_slice comes from &a[i_row_block_start..], so its 0th element
        // is A(i_row_block_start, pc_of_strip). We advance by columns of the original A matrix.
        let source_col_start_offset = p_col_in_panel * m_original_matrix;

        // Offset in destination for the current packed column segment.
        let dest_col_segment_start_offset = p_col_in_panel * MR;

        if mr_effective_in_panel > 0 {
            unsafe {
                let src_ptr = a_panel_source_slice.as_ptr().add(source_col_start_offset);
                let dest_ptr = dest_slice.as_mut_ptr().add(dest_col_segment_start_offset);

                // The `mr_effective_in_panel` elements from this column of A are contiguous.
                // Copy them in one go. The rest of the MR-sized dest segment is already zero.
                copy_nonoverlapping(src_ptr, dest_ptr, mr_effective_in_panel);
            }
        }
        // If mr_effective_in_panel == 0, this MR-sized segment in dest_slice remains all zeros.
        // If mr_effective_in_panel < MR, the tail of this MR-sized segment remains all zeros.
    }
}

/// Packs a panel of matrix B (column-major) into dest_slice.
/// Output format is row-by-row of the B-panel.
/// Relies on dest_slice being pre-zeroed for padding.
#[inline(always)]
fn pack_panel_b(
    dest_slice: &mut [f32],       // Pre-zeroed, length kc_panel * NR
    b_panel_source_slice: &[f32], // Points to B(pc_panel_start, jc_panel_start for this panel)
    nr_effective_in_panel: usize, // <= NR, number of cols in B's panel to pack
    kc_panel: usize,              // Number of rows in B's panel to pack (K-dim)
    k_original_matrix: usize,     // Leading dimension of original matrix B
) {
    debug_assert_eq!(dest_slice.len(), kc_panel * NR, "Dest B slice len mismatch");
    debug_assert!(nr_effective_in_panel <= NR, "nr_eff_b > NR");

    // dest_slice is assumed to be pre-zeroed.

    for p_row_in_panel in 0..kc_panel {
        // Iterate over "rows" of B's panel (K-dimension)
        let dest_row_start_offset = p_row_in_panel * NR;

        // For each "row" of the B-panel, the source elements are strided in col-major B.
        // Copy nr_effective_in_panel elements element by element (scalar gather).
        // The remaining (NR - nr_effective_in_panel) elements in dest_slice for this packed row
        // are already zero.
        for j_col_in_panel in 0..nr_effective_in_panel {
            // Iterate over "columns" of B's panel
            let source_index = j_col_in_panel * k_original_matrix + p_row_in_panel;
            dest_slice[dest_row_start_offset + j_col_in_panel] = b_panel_source_slice[source_index];
        }
        // If nr_effective_in_panel == 0, this NR-sized segment in dest_slice remains all zeros.
    }
}

// --- Block Packing Functions ---

/// Packs a block of matrix A (column-major).
#[inline(always)]
fn pack_block_a(
    a_block_source_slice: &[f32], // Slice representing A(ic..ic+mc-1, pc..pc+kc-1)
    // Starts at memory of A(ic, pc)
    mc_block: usize,          // Total number of rows in this block of A
    kc_block: usize,          // Total number of columns (depth) in this block of A
    m_original_matrix: usize, // Leading dimension of original matrix A
) -> Vec<f32> {
    if mc_block == 0 || kc_block == 0 {
        return Vec::new();
    }

    let num_row_panels = mc_block.msrv_div_ceil(MR);
    let total_packed_size = num_row_panels * kc_block * MR;

    let mut packed_block_a_data = alloc_zeroed_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT);
    let mut current_write_offset_in_packed = 0;

    // Iterate over the mc_block rows in MR-sized steps
    for i_row_panel_start_in_block in (0..mc_block).step_by(MR) {
        let mr_effective_for_panel = min(MR, mc_block - i_row_panel_start_in_block);

        // The source slice for pack_panel_a_into should point to
        // A(ic_block + i_row_panel_start_in_block, pc_block).
        // a_block_source_slice points to A(ic_block, pc_block).
        // So, we need to offset by i_row_panel_start_in_block elements down the first column
        // of a_block_source_slice.
        let panel_source_slice = &a_block_source_slice[i_row_panel_start_in_block..];

        let dest_slice_len_for_this_panel = kc_block * MR; // Each panel packs into kc_block * MR
        let dest_sub_slice = &mut packed_block_a_data[current_write_offset_in_packed
            ..current_write_offset_in_packed + dest_slice_len_for_this_panel];

        pack_panel_a_into(
            dest_sub_slice,
            panel_source_slice,
            mr_effective_for_panel,
            kc_block, // The "kc_panel" for pack_panel_a_into is the full kc_block
            m_original_matrix,
        );
        current_write_offset_in_packed += dest_slice_len_for_this_panel;
    }

    packed_block_a_data
}

/// Packs a block of matrix B (column-major).
#[inline(always)]
fn pack_block_b(
    b_block_source_slice: &[f32], // Slice representing B(pc..pc+kc-1, jc..jc+nc-1)
    // Starts at memory of B(pc, jc)
    nc_block: usize,          // Total number of columns in this block of B
    kc_block: usize,          // Total number of rows (depth) in this block of B
    k_original_matrix: usize, // Leading dimension of original matrix B
) -> Vec<f32> {
    if kc_block == 0 || nc_block == 0 {
        return Vec::new();
    }

    let num_col_panels = nc_block.msrv_div_ceil(NR);
    let total_packed_size = num_col_panels * kc_block * NR;

    let mut packed_block_b_data = alloc_zeroed_f32_vec(total_packed_size, f32x8::AVX_ALIGNMENT);
    let mut current_write_offset_in_packed = 0;

    // Iterate over the nc_block columns in NR-sized steps
    for j_col_panel_start_in_block in (0..nc_block).step_by(NR) {
        let nr_effective_for_panel = min(NR, nc_block - j_col_panel_start_in_block);

        // The source slice for pack_panel_b should point to
        // B(pc_block, jc_block + j_col_panel_start_in_block).
        // b_block_source_slice points to B(pc_block, jc_block).
        // So, we need to offset by j_col_panel_start_in_block columns.
        // In column-major, this is an offset of j_col_panel_start_in_block * k_original_matrix elements.
        let panel_source_data_start_offset = j_col_panel_start_in_block * k_original_matrix;
        let current_panel_source_slice = &b_block_source_slice[panel_source_data_start_offset..];

        let dest_slice_len_for_this_panel = kc_block * NR; // Each panel packs into kc_block * NR
        let dest_sub_slice = &mut packed_block_b_data[current_write_offset_in_packed
            ..current_write_offset_in_packed + dest_slice_len_for_this_panel];

        pack_panel_b(
            dest_sub_slice,
            current_panel_source_slice,
            nr_effective_for_panel,
            kc_block, // The "kc_panel" for pack_panel_b is the full kc_block
            k_original_matrix,
        );
        current_write_offset_in_packed += dest_slice_len_for_this_panel;
    }

    packed_block_b_data
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
#[target_feature(enable = "avx2,avx,fma")]
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
                                        kernel_8x6(
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
#[target_feature(enable = "avx2,avx,fma")]
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

                                        kernel_8x6(
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
