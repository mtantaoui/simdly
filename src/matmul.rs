use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::cmp::min;

use crate::{
    simd::{avx2::f32x8::F32x8, traits::SimdVec},
    KC, MC, MR, NC, NR,
};

/// Packs a panel of matrix B into a contiguous buffer with a specific layout optimized for the kernel.
/// This version initializes the entire packed buffer with zeros upfront and then copies
/// data, which can be more performant by leveraging optimized memory initialization
/// and avoiding per-row padding logic.
///
/// Matrix B is assumed to be in column-major format.
///
/// ## Packed Format Description
///
/// The panel from B is packed "row by row" from the perspective of the panel itself.
/// Specifically:
/// 1. The resulting buffer `panel_b_packed` has dimensions `kc * NR`.
/// 2. It iterates `p` from `0` to `kc-1` (representing rows of the panel, effectively slices along the K-dimension).
/// 3. For each `p`, it copies `nr` elements `B(pc+p, jc+j_panel)` for `j_panel` from `0` to `nr-1`
///    into `panel_b_packed`.
/// 4. These `nr` elements are followed by `NR - nr` zeros to pad the packed row to `NR` elements.
/// 5. This process is repeated for all `kc` "rows" of the panel, laid out contiguously.
///
/// The memory layout will be:
/// ```
/// // p = 0: [ B(pc+0, jc+0), B(pc+0, jc+1), ..., B(pc+0, jc+nr-1), 0.0, ..., 0.0 ] (NR elements total)
/// // p = 1: [ B(pc+1, jc+0), B(pc+1, jc+1), ..., B(pc+1, jc+nr-1), 0.0, ..., 0.0 ] (NR elements total)
/// // ...
/// // p = kc-1: [ B(pc+kc-1, jc+0), ..., B(pc+kc-1, jc+nr-1), 0.0, ..., 0.0 ] (NR elements total)
/// ```
/// This layout is effectively row-major storage of the `kc x nr` panel, where each row is padded to `NR`.
/// It is suitable for micro-kernels that consume "rows" of B (where a "row" of B here means elements
/// `B(k_idx, n_idx_start..n_idx_end)` for a fixed `k_idx`).
///
/// # Arguments
///
/// * `b` - A slice into the original column-major matrix B's data. This slice must
///   start at the memory location corresponding to element `B(pc, jc)` of the original matrix B,
///   where `(pc, jc)` is the top-left corner of the panel being packed.
/// * `nr` - The number of actual columns to pack from this panel of B. Must satisfy `nr <= NR`.
/// * `kc` - The number of "rows" (or depth along K-dimension) to pack from this panel of B.
/// * `k` - The leading dimension (total number of rows) of the original, full matrix B.
///   This is used to correctly step between columns in the input slice `b`.
/// * `NR` - The packing dimension for columns. Each "row" of `kc` elements in the packed
///   panel will be padded with zeros to this width. This typically matches SIMD vector width
///   or a multiple thereof.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed panel of B. The total size of the vector will be `kc * NR`.
///
/// # Panics
///
/// Panics if `nr > NR`.
fn pack_panel_b(b: &[f32], nr: usize, kc: usize, k: usize) -> Vec<f32> {
    // Ensure that the number of actual columns to pack (`nr`) does not exceed
    // the target packing width (`NR`). Padding only makes sense if nr <= NR.
    assert!(
        nr <= NR,
        "nr (actual columns: {}) cannot exceed NR (target packed width: {})",
        nr,
        NR
    );

    // Allocate the entire packed buffer and initialize it with zeros.
    // The total size is kc "rows", each of length NR.
    // This upfront zeroing handles all necessary padding implicitly.
    // For many systems, `vec![0.0; size]` can be heavily optimized (e.g., using memset).
    let mut panel_b_packed = vec![0.0f32; kc * NR];

    // Iterate over the `kc` "rows" of the panel. Each `p` corresponds to a
    // specific K-index offset (pc+p) in the original matrix B.
    for p in 0..kc {
        // Calculate the starting index in `panel_b_packed` for the current "row" `p`.
        // Each packed "row" occupies `NR` elements.
        let packed_row_start_index = p * NR;

        // Iterate over the `nr` actual columns of the panel B that we need to copy.
        // `j` here corresponds to the column index within the sub-panel (jc+j in original matrix).
        for j in 0..nr {
            // Accessing B(pc+p, jc+j) from the input slice `b`.
            // The input slice `b` starts at the memory location of B(pc, jc) in the
            // original column-major matrix B.
            // In a column-major matrix with leading dimension `k`, the element at
            // row `row_idx` and column `col_idx` is at `memory[col_idx * k + row_idx]`.
            // So, B(pc+p, jc+j) would be at global memory offset `(jc+j)*k + (pc+p)`.
            // Since `b` is a slice starting at `B(pc, jc)` (global offset `jc*k + pc`),
            // the element B(pc+p, jc+j) is found at index `((jc+j)*k + (pc+p)) - (jc*k + pc)`
            // within the slice `b`, which simplifies to `j*k + p`.
            let source_index = j * k + p;

            // Copy the element from the source panel `b` to the destination packed panel.
            panel_b_packed[packed_row_start_index + j] = b[source_index];
        }
        // Elements from `panel_b_packed[packed_row_start_index + nr]` to
        // `panel_b_packed[packed_row_start_index + NR - 1]` remain 0.0
        // due to the initial `vec![0.0f32; kc * NR]` initialization.
        // No explicit padding loop is needed here for each row `p`.
    }

    panel_b_packed
}

/// Packs a block of matrix B by iteratively packing NR-column-wide panels.
/// Matrix B is assumed to be in column-major format.
///
/// # Arguments
///
/// * `b` - A slice representing a block of matrix B, `B(pc..pc+kc-1, jc..jc+nc-1)`.
/// * `nc` - The total number of columns in this block of B.
/// * `kc` - The total number of rows (depth) in this block of B.
/// * `k` - The leading dimension (number of rows) of the original matrix B.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed block of B.
fn pack_block_b(b: &[f32], nc: usize, kc: usize, k: usize) -> Vec<f32> {
    let mut block_b_packed = Vec::with_capacity(nc * kc); // Rough estimate, actual might differ due to NR padding

    for j in (0..nc).step_by(NR) {
        // Iterate over columns of B_block in NR-sized steps
        let nr_effective = min(NR, nc - j); // Number of columns for the current panel (can be < NR at the edge)
                                            // `&b[j * k..]` advances the slice `b` to point to the j-th column (in this block's view)
                                            // This corresponds to `B(pc, jc+j)` in original matrix terms.
        let panel_b_packed = pack_panel_b(&b[j * k..], nr_effective, kc, k);
        block_b_packed.extend(panel_b_packed);
    }

    block_b_packed
}

/// Packs a panel of matrix A into a contiguous buffer with a specific layout.
/// Matrix A is assumed to be in column-major format.
/// The packed format is row-major within each `MR`-element row vector (transposed from A's column),
/// and these vectors are laid out sequentially for `kc` depth.
/// Specifically, for each `p` from `0..kc`, it packs `mr` elements `A[i, p]` (original indexing),
/// followed by padding to `MR`.
///
/// # Arguments
///
/// * `a` - A slice representing a sub-panel of matrix A, starting at `A(ic, pc)`.
/// * `mr` - The number of rows to pack from this panel of A (actual `mr <= MR`).
/// * `kc` - The number of columns to pack from this panel of A (depth, `kc <= KC`).
/// * `m` - The leading dimension (number of rows) of the original matrix A.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed panel of A. The size is `kc * MR`.
fn pack_panel_a(a: &[f32], mr: usize, kc: usize, m: usize) -> Vec<f32> {
    // The resulting packed vector will store elements for SIMD processing.
    // Layout: [A_i0_p0, A_i1_p0, ..., A_i(mr-1)_p0, <pad_to_MR>
    //          A_i0_p1, A_i1_p1, ..., A_i(mr-1)_p1, <pad_to_MR>
    //          ...
    //          A_i0_p(kc-1), ..., A_i(mr-1)_p(kc-1), <pad_to_MR>]
    // Each group of MR elements corresponds to one `p` index (a column vector from A, optimized for SIMD row access).
    let mut panel_a_packed = Vec::with_capacity(kc * MR);

    // let mut panel_a_packed = aligned_vec_f32(kc * MR, 32);

    for p in 0..kc {
        // Iterate over the K-dimension (columns of the panel)
        for i in 0..mr {
            // Iterate over the M-dimension (rows of the panel)
            // Access A[i, p] from the perspective of the input sub-panel `a`.
            // `a` starts at `A(ic, pc)`, so `a[p * m + i]` accesses `A(ic+i, pc+p)`.
            panel_a_packed.push(a[p * m + i]);
        }
        // Pad with zeros if `mr` (actual rows in this panel) is less than `MR` (SIMD register width for rows)
        panel_a_packed.extend(vec![0.0; MR - mr]);
    }

    panel_a_packed
}

/// Packs a block of matrix A by iteratively packing MR-row-high panels.
/// Matrix A is assumed to be in column-major format.
///
/// # Arguments
///
/// * `a` - A slice representing a block of matrix A, `A(ic..ic+mc-1, pc..pc+kc-1)`.
/// * `mc` - The total number of rows in this block of A.
/// * `kc` - The total number of columns (depth) in this block of A.
/// * `m` - The leading dimension (number of rows) of the original matrix A.
///
/// # Returns
///
/// A `Vec<f32>` containing the packed block of A.
fn pack_block_a(a: &[f32], mc: usize, kc: usize, m: usize) -> Vec<f32> {
    let mut block_a_packed = Vec::with_capacity(mc * kc); // Rough estimate, actual might differ due to MR padding

    for i in (0..mc).step_by(MR) {
        // Iterate over rows of A_block in MR-sized steps
        let mr_effective = min(MR, mc - i); // Number of rows for the current panel (can be < MR at the edge)
                                            // `&a[i..]` advances the slice `a` to point to the i-th row (in this block's view).
                                            // This corresponds to `A(ic_block + i, pc_block)` in original matrix terms.
        let panel_a_packed = pack_panel_a(&a[i..], mr_effective, kc, m);
        block_a_packed.extend(panel_a_packed);
    }

    block_a_packed
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
                                        kernel_MRxNR(
                                            a_panel,
                                            b_panel,
                                            c_micropanel,
                                            mr_eff,
                                            nr_eff,
                                            kc,
                                            m, // This is ldc (leading dim of C)
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

                                        kernel_MRxNR(
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
    debug_assert_eq!(MR, 8);
    debug_assert!(
        mr_eff <= MR,
        "mr_eff ({}) must be less than or equal to MR ({})",
        mr_eff,
        MR
    );
    debug_assert!(
        nr_eff <= NR,
        "nr_eff ({}) must be less than or equal to NR ({})",
        nr_eff,
        NR
    );

    let mut c_simd_cols: [F32x8; NR] = [unsafe { F32x8::splat(0.0) }; NR]; // Max NR cols

    // Load C columns into SIMD registers
    for j_col in 0..nr_eff {
        let c_col_ptr = unsafe { c_micro_panel.as_ptr().add(j_col * ldc) };
        c_simd_cols[j_col] = match mr_eff.cmp(&MR) {
            std::cmp::Ordering::Equal => unsafe { F32x8::load(c_col_ptr, mr_eff) },
            std::cmp::Ordering::Less => unsafe { F32x8::load_partial(c_col_ptr, mr_eff) },
            std::cmp::Ordering::Greater => panic!("mr_eff > MR"),
        };
    }

    for p in 0..kc {
        // Loop over K dimension
        let a_simd_vec = F32x8::new(&a_panel[p * MR..p * MR + MR]);

        for j_col in 0..nr_eff {
            // Loop over N dimension (columns of B and C micro-panel)
            let b_scalar = b_panel[p * NR + j_col]; // Access correct B element for current column
            let b_simd_splat = unsafe { F32x8::splat(b_scalar) };
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

    #[test]
    fn test_pack_panel_b() {
        // B is 3x4 (k_orig=3, n=4)
        // 1  4  7 10
        // 2  5  8 11
        // 3  6  9 12
        // Sub-panel: B(0..kc-1, 0..NR-1)
        // Let NR_test = 2, kc_test = 2. Panel is B(0..1, 0..1)
        // Slice b_sub: [1,2,3, 4,5,6] (col-major view of B(0..2,0..1))
        // Original k=3
        let k_orig = 3;
        let b_orig_data = [
            1.0, 2.0, 3.0, // col 0
            4.0, 5.0, 6.0, // col 1
            7.0, 8.0, 9.0, // col 2
            10.0, 11.0, 12.0, // col 3
        ];

        // Test case 1: nr_eff == NR, kc = 2
        // Panel B(0..1, 0..NR-1)
        let panel_b_slice = &b_orig_data[at(0, 0, k_orig)..]; // B(0,0) onwards
        let packed_b = pack_panel_b(panel_b_slice, NR, 2, k_orig);
        // Expected layout: [B_00, B_01, ..., B_0(NR-1), (pad)]
        //                  [B_10, B_11, ..., B_1(NR-1), (pad)]
        // Here global NR = 4. Actual nr_eff for this call is 4.
        // kc = 2.
        // B(0,0)=1, B(0,1)=4, B(0,2)=7, B(0,3)=10
        // B(1,0)=2, B(1,1)=5, B(1,2)=8, B(1,3)=11
        let mut expected_b1 = vec![0.0; NR * 2];
        for j in 0..NR {
            // p=0
            if j < 4 {
                expected_b1[j] = b_orig_data[at(0, j, k_orig)];
            } // B(0,j)
        }
        for j in 0..NR {
            // p=1
            if j < 4 {
                expected_b1[NR + j] = b_orig_data[at(1, j, k_orig)];
            } // B(1,j)
        }
        assert_eq!(packed_b.len(), NR * 2);
        assert_eq!(packed_b, expected_b1);

        // Test case 2: nr_eff < NR (e.g. nr_eff = 1), kc = 3
        let nr_eff_test = 1;
        let packed_b_partial = pack_panel_b(panel_b_slice, nr_eff_test, 3, k_orig);
        let mut expected_b2 = vec![0.0; NR * 3]; // Padded to NR
        for p_val in 0..3 {
            // kc = 3
            for j in 0..nr_eff_test {
                expected_b2[p_val * NR + j] = b_orig_data[at(p_val, j, k_orig)];
                // B(p,j)
            }
        }
        assert_eq!(packed_b_partial.len(), NR * 3);
        assert_eq!(packed_b_partial, expected_b2);
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
        let mut c_data = vec![0.0; m * n];

        let expected_c = naive_matmul(&a_data, &b_data, m, n, k);

        if use_par {
            par_matmul(&a_data, &b_data, &mut c_data, m, n, k);
        } else {
            // To test matmul with the *fixed* kernel, you'd need to modify matmul
            // to call kernel_MRxNR_fixed. For now, testing with original kernel_8x1.
            matmul(&a_data, &b_data, &mut c_data, m, n, k);
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
        let mut c_data = vec![0.0; m * n];

        // Using original matmul with potentially buggy kernel
        matmul(&a_data, &b_data, &mut c_data, m, n, k);

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
