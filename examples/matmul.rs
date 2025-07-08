//! # High-Performance Matrix Multiplication (GEMM)
//!
//! This module provides a highly optimized implementation of the general matrix-matrix
//! multiplication operation, `C = A * B`, for single-precision floating-point numbers (`f32`).
//!
//! The implementation is based on the BLIS/GotoBLAS algorithm, which achieves high
//! performance by:
//! 1.  **Cache-Aware Blocking:** The matrices are partitioned into blocks that fit within
//!     different levels of the CPU cache hierarchy (L1, L2, L3). This minimizes data
//!     movement between RAM and the CPU.
//! 2.  **Data Packing:** Blocks of matrices `A` and `B` are copied ("packed") into
//!     contiguous, aligned memory buffers. This has two major benefits:
//!     - It ensures that data access within the core computation is sequential, which is
//!       optimal for hardware prefetchers.
//!     - It allows for the use of aligned SIMD (Single Instruction, Multiple Data) loads,
//!       which are more efficient.
//! 3.  **Optimized Micro-Kernel:** A highly tuned "micro-kernel" performs the multiplication
//!     of small, packed "panels". This kernel is written using AVX2 intrinsics to exploit
//!     data-level parallelism, performing multiple floating-point operations simultaneously.
//!
//! The matrices are assumed to be in **column-major** order, which is standard in Fortran
//! and libraries like BLAS and LAPACK.

use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::traits::SimdVec;
use simdly::simd::utils::alloc_zeroed_f32_vec;

// --- Algorithm Constants ---

/// The memory alignment required for AVX2 SIMD instructions. AVX instructions operate
/// on 256-bit (32-byte) registers, so aligning memory to 32-byte boundaries
/// allows for the use of faster, aligned load/store instructions.
const ALIGNMENT: usize = 32;

/// Micro-kernel dimension: Register block size for rows of A and C.
/// This corresponds to the number of rows of C that can be computed with vector registers
/// in the micro-kernel. For `f32` on AVX2, which has 256-bit registers, `8` is a natural fit (8 * 32 bits = 256 bits).
pub const MR: usize = 8;

/// Micro-kernel dimension: Register block size for columns of B and C.
/// This corresponds to the number of columns of C that can be computed simultaneously.
/// This is typically tuned based on the number of available vector registers.
pub const NR: usize = 8;

/// Cache-level dimension: Block size for the K dimension (inner dimension of the multiplication).
/// This is tuned to ensure that a packed `KC x NR` panel of B and an `MR x KC` panel of A
/// can fit comfortably in the L1 or L2 cache.
pub const KC: usize = 16;

/// Cache-level dimension: Block size for rows of A and C.
/// This is a multiple of `MR` and is tuned for L2/L3 cache performance.
pub const MC: usize = 16; // Example value, should be a multiple of MR

/// Cache-level dimension: Block size for columns of B and C.
/// This is a multiple of `NR` and is tuned for L3 cache performance.
pub const NC: usize = 16; // Example value, should be a multiple of NR

// --- Helper Functions ---

/// Calculates the 1D index for a 2D element in a column-major matrix.
///
/// # Arguments
/// * `i` - Row index.
/// * `j` - Column index.
/// * `ld` - Leading dimension (number of rows in the matrix).
#[inline(always)]
fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

/// Utility function to print a matrix stored in column-major format.
pub fn display_matrix_column_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
    println!("---");
}

/// Utility function to print a matrix (for debugging row-major packed data).
pub fn display_matrix_row_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for j in 0..n {
        for i in 0..m {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
    println!("---");
}

// --- Packed Data Structures for Matrix B ---

/// Represents a single packed panel of the `B` matrix, with dimensions `KC x NR`.
///
/// The data is stored in a **row-major** format within the panel. This means each of the `KC` rows
/// contains `NR` floating-point numbers. This layout is optimal for the micro-kernel,
/// which consumes one row of this panel at a time and broadcasts its elements.
///
/// The `#[repr(C, align(32))]` attribute is critical. It ensures that each `BPanel` instance
/// is aligned to a 32-byte boundary, satisfying the alignment requirements for AVX2.
#[repr(C, align(32))]
pub struct BPanel<const KC: usize, const NR: usize> {
    pub data: [[f32; NR]; KC],
}

/// Represents a packed block of the `B` matrix, composed of multiple `BPanel`s.
///
/// This struct manages a contiguous, 32-byte aligned memory allocation that holds
/// a sequence of `BPanel` structs. It handles the raw memory allocation and ensures
/// safe deallocation via its `Drop` implementation.
pub struct BBlock<const KC: usize, const NR: usize> {
    /// A raw pointer to the heap-allocated, aligned block of `BPanel`s.
    ptr: *mut BPanel<KC, NR>,
    /// The number of `BPanel`s allocated in the memory block.
    num_panels: usize,
    /// The memory `Layout` used for allocation. Storing this guarantees that
    /// deallocation is performed with the exact same layout, which is required for safety.
    layout: Layout,
    /// The original number of columns from matrix `B` packed into this block.
    pub nc: usize,
    /// `PhantomData` informs the Rust compiler that this struct "owns" the data
    /// pointed to by `ptr`, enabling proper borrow checking and drop semantics.
    _marker: PhantomData<BPanel<KC, NR>>,
}

impl<const KC: usize, const NR: usize> BBlock<KC, NR> {
    /// Allocates aligned, zero-initialized memory for a packed block of `B`.
    ///
    /// # Arguments
    /// * `nc`: The number of columns from the original `B` matrix to be packed in this block.
    ///
    /// # Returns
    /// A `Result` containing the new `BBlock` on success, or a `Layout` error on failure.
    pub fn new(nc: usize) -> Result<Self, Layout> {
        // Calculate how many NR-wide panels are needed to store `nc` columns.
        let num_panels = nc.div_ceil(NR);

        // Define the memory layout for an array of `BPanel`s.
        let layout = Layout::array::<BPanel<KC, NR>>(num_panels).unwrap();

        // Ensure the layout meets our minimum alignment requirement.
        let layout = layout.align_to(ALIGNMENT).unwrap();

        let ptr = unsafe {
            // Allocate zero-initialized memory. `alloc_zeroed` is used because packing
            // may not fill every element (due to edge cases), and zero-padding is correct.
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<BPanel<KC, NR>>()
        };

        Ok(BBlock {
            ptr,
            num_panels,
            layout,
            nc,
            _marker: PhantomData,
        })
    }

    /// Returns an immutable slice view of the `BPanel`s.
    pub fn as_panels(&self) -> &[BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `BPanel`s.
    pub fn as_panels_mut(&mut self) -> &mut [BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// The `Drop` implementation ensures that the heap-allocated memory is safely
/// deallocated when a `BBlock` goes out of scope.
impl<const KC: usize, const NR: usize> Drop for BBlock<KC, NR> {
    fn drop(&mut self) {
        // Deallocating with a zero-sized layout is undefined behavior.
        // This check prevents UB if a BBlock was created with `nc = 0`.
        if self.layout.size() > 0 {
            unsafe {
                // Deallocate the memory using the exact layout that was stored
                // during allocation. This is the only safe way to deallocate
                // memory obtained from the global allocator.
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

// --- Convenience Indexing for BBlock ---

impl<const KC: usize, const NR: usize> Index<usize> for BBlock<KC, NR> {
    type Output = BPanel<KC, NR>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const KC: usize, const NR: usize> IndexMut<usize> for BBlock<KC, NR> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Packed Data Structures for Matrix A ---

/// Represents a single packed panel of the `A` matrix, with dimensions `KC x MR`.
///
/// The data is stored in a **column-major** format within the panel. This means each of the `KC`
/// "rows" of this struct actually holds one column of length `MR`. This layout is optimal
/// for the micro-kernel, which needs to load `MR`-element columns of `A` into vector registers.
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for efficient AVX2 loads.
#[repr(C, align(32))]
pub struct APanel<const MR: usize, const KC: usize> {
    pub data: [[f32; MR]; KC],
}

/// Represents a packed block of the `A` matrix, composed of multiple `APanel`s.
///
/// This struct is analogous to `BBlock` but is designed for matrix `A`. It manages
/// the aligned, heap-allocated memory for the packed `A` data.
pub struct ABlock<const MR: usize, const KC: usize> {
    /// A raw pointer to the heap-allocated, aligned block of `APanel`s.
    ptr: *mut APanel<MR, KC>,
    /// The number of `APanel`s allocated in the memory block.
    num_panels: usize,
    /// The memory `Layout` used for allocation, required for safe deallocation.
    layout: Layout,
    /// The original number of rows from matrix `A` packed into this block.
    pub mc: usize,
    /// `PhantomData` for ownership and drop-check semantics.
    _marker: PhantomData<APanel<MR, KC>>,
}

impl<const MR: usize, const KC: usize> ABlock<MR, KC> {
    /// Allocates aligned, zero-initialized memory for a packed block of `A`.
    pub fn new(mc: usize) -> Result<Self, Layout> {
        let num_panels = mc.div_ceil(MR);

        let layout = Layout::array::<APanel<MR, KC>>(num_panels)
            .expect("Invalid layout for APanel")
            .align_to(ALIGNMENT)
            .unwrap();

        let ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<APanel<MR, KC>>()
        };

        Ok(ABlock {
            ptr,
            num_panels,
            layout,
            mc,
            _marker: PhantomData,
        })
    }

    /// Returns an immutable slice view of the `APanel`s.
    pub fn as_panels(&self) -> &[APanel<MR, KC>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `APanel`s.
    pub fn as_panels_mut(&mut self) -> &mut [APanel<MR, KC>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// Safe deallocation for `ABlock`'s memory.
impl<const MR: usize, const KC: usize> Drop for ABlock<MR, KC> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

/// Convenience indexing for `ABlock`.
impl<const MR: usize, const KC: usize> Index<usize> for ABlock<MR, KC> {
    type Output = APanel<MR, KC>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const MR: usize, const KC: usize> IndexMut<usize> for ABlock<MR, KC> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Core Matrix Multiplication Logic ---

/// Computes `C = A * B` using a cache-blocked, packing-based algorithm.
///
/// This function implements the "macro-kernel" of the GEMM operation. It iterates
/// through the matrices `A`, `B`, and `C` in blocks of size `MC x KC`, `KC x NC`,
/// and `MC x NC` respectively.
///
/// # Arguments
/// * `a`: Slice containing matrix A in column-major order.
/// * `b`: Slice containing matrix B in column-major order.
/// * `c`: Mutable slice for the output matrix C, in column-major order. Assumed to be zero-initialized.
/// * `m`: Number of rows in A and C.
/// * `n`: Number of columns in B and C.
/// * `k`: Number of columns in A and rows in B.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // The outer-most loop iterates over columns of C and B in blocks of size NC.
    // This is the "j_c" loop.
    for jc in (0..n).step_by(NC) {
        let nc = min(NC, n - jc);

        // This loop iterates over the K dimension in blocks of size KC.
        // This is the "p_c" loop. It is placed here instead of inside the `ic` loop
        // to promote keeping the packed block of B in a higher-level cache (e.g., L3)
        // while it is reused across multiple blocks of A.
        for pc in (0..k).step_by(KC) {
            let kc = min(KC, k - pc);

            // Pack a `kc x nc` block of B starting at `(pc, jc)`.
            // The resulting `b_block` is organized for efficient access by the micro-kernel.
            let b_block = pack_b::<KC, NR>(b, nc, kc, k, pc, jc);

            // This loop iterates over rows of C and A in blocks of size MC.
            // This is the "i_c" loop.
            for ic in (0..m).step_by(MC) {
                let mc = min(MC, m - ic);

                // Pack a `mc x kc` block of A starting at `(ic, pc)`.
                // The resulting `a_block` is organized for efficient access.
                let a_block = pack_a::<MR, KC>(a, mc, kc, m, ic, pc);

                // --- Micro-Kernel Execution ---
                // The packed blocks of A and B are now fed to the micro-kernel.
                // The loops iterate over `NR`-wide panels of B and `MR`-wide panels of A.
                for jr in 0..b_block.as_panels().len() {
                    let b_panel = &b_block[jr];
                    let nr = min(NR, nc - jr * NR);

                    for ir in 0..a_block.as_panels().len() {
                        let a_panel = &a_block[ir];
                        let mr = min(MR, mc - ir * MR);

                        // Calculate the top-left corner of the target `mr x nr` micro-panel in C.
                        let c_row = ic + ir * MR;
                        let c_col = jc + jr * NR;
                        let c_micropanel_start_idx = at(c_row, c_col, m);

                        let c_micropanel = &mut c[c_micropanel_start_idx..];

                        unsafe {
                            kernel_MRxNR(a_panel, b_panel, c_micropanel.as_mut_ptr(), mr, nr, kc, m)
                        };

                        // // PRODUCTION NOTE: The current `kernel_8x8` assumes `nr == NR`.
                        // // A production-ready version must handle edge cases where `nr < NR`.
                        // // This is a critical safety and correctness issue.
                        // // See the `kernel_8x8` documentation for details.
                        // if mr == MR && nr == NR {
                        //     unsafe {
                        //         kernel_8x8(
                        //             a_panel,
                        //             b_panel,
                        //             c_micropanel.as_mut_ptr(),
                        //             mr,
                        //             nr,
                        //             kc,
                        //             m,
                        //         );
                        //     }
                        // } else {
                        //     // A general    kernel for edge cases (mr < MR or nr < NR) would be called here.
                        //     // For simplicity, this example only includes the optimized kernel.
                        //     // We will call the main kernel but this is UNSAFE if nr < NR.
                        //     // A safe implementation would use a different kernel or scalar code.
                        //     unsafe {
                        //         kernel_8x8(
                        //             a_panel,
                        //             b_panel,
                        //             c_micropanel.as_mut_ptr(),
                        //             mr,
                        //             nr,
                        //             kc,
                        //             m,
                        //         );
                        //     }
                        // }
                    }
                }
            }
        }
    }
}

/// Packs a block from the column-major matrix `A` into a specialized, contiguous `ABlock`.
///
/// The destination `ABlock` stores data in a column-major format within its panels,
/// optimized for loading entire `MR`-element columns into SIMD registers.
///
/// # Arguments
/// * `a`: The source matrix A (column-major).
/// * `mc`, `kc`: The dimensions of the block to pack.
/// * `m`: The leading dimension (total number of rows) of the source matrix `a`.
/// * `ic`, `pc`: The row and column coordinates of the top-left corner of the block in `a`.
pub fn pack_a<const MR: usize, const KC: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,
    ic: usize,
    pc: usize,
) -> ABlock<MR, KC> {
    let mut packed_block = ABlock::<MR, KC>::new(mc).expect("Memory allocation failed for ABlock");

    // Iterate over the `mc` rows of the block in `MR`-high row-panels.
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let mr_in_panel = min(MR, mc - i_panel_start); // Handle the bottom edge case.

        // Iterate through the `kc` columns of the block.
        for p_col in 0..kc {
            // Identify the source column and row in the original matrix A.
            let src_col = pc + p_col;
            let src_row_start = ic + i_panel_start;

            // In column-major A, the desired column `src_col` is a contiguous slice.
            // Calculate the start of this slice.
            let src_start = at(src_row_start, src_col, m);
            let src_slice = &a[src_start..src_start + mr_in_panel];

            // The destination in the panel is also a contiguous slice.
            let dest_slice = &mut dest_panel.data[p_col][0..mr_in_panel];

            // Copy the column from A into the packed panel.
            dest_slice.copy_from_slice(src_slice);
        }
    }

    packed_block
}

/// Packs a block from the column-major matrix `B` into a specialized, contiguous `BBlock`.
///
/// The destination `BBlock` stores data in a row-major format within its panels. This
/// layout is designed so the micro-kernel can read one row of the packed panel and
/// broadcast its elements to SIMD registers.
///
/// # Arguments
/// * `b`: The source matrix B (column-major).
/// * `nc`, `kc`: The dimensions of the block to pack.
/// * `k`: The leading dimension (total number of rows) of the source matrix `b`.
/// * `pc`, `jc`: The row and column coordinates of the top-left corner of the block in `b`.
pub fn pack_b<const KC: usize, const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,
    pc: usize,
    jc: usize,
) -> BBlock<KC, NR> {
    let mut packed_block = BBlock::<KC, NR>::new(nc).expect("Memory allocation failed for BBlock");

    // Iterate over the `nc` columns of the block in `NR`-wide column-panels.
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let nr_in_panel = min(NR, nc - j_panel_start); // Handle the right edge case.

        // Fill the destination panel row by row. Each row in the panel corresponds
        // to a row from the source block of B.
        for p_row in 0..kc {
            // Identify the source row in the original matrix B.
            let src_row = pc + p_row;

            // Copy elements from different columns of B to form a contiguous row in the panel.
            for j_col_in_panel in 0..nr_in_panel {
                // Identify the source column in the original matrix B.
                let src_col = jc + j_panel_start + j_col_in_panel;

                // In column-major B, `B(r,c)` is at `c*ld + r`.
                let src_idx = at(src_row, src_col, k);

                // Store the element in its row-major position in the destination panel.
                dest_panel.data[p_row][j_col_in_panel] = b[src_idx];
            }
        }
    }

    packed_block
}

/// The AVX2-accelerated 8x8 micro-kernel for `C += A * B`.
///
/// This function computes the matrix product on small, packed panels:
/// - An `mr x kc` panel from `A`.
/// - A `kc x nr` panel from `B`.
/// - It updates an `mr x nr` micro-panel of `C`.
///
/// It uses 8 AVX2 vector registers to hold an 8x8 sub-block of `C`, accumulating
/// results via fused multiply-add (FMA) instructions.
///
/// # Arguments
/// * `a_panel`: The packed panel from `A`, with data stored column-wise.
/// * `b_panel`: The packed panel from `B`, with data stored row-wise.
/// * `c_micropanel`: A raw pointer to the top-left element of the `mr x nr` destination in C.
/// * `mr`, `nr`, `kc`: The actual dimensions of the operation.
/// * `m`: The leading dimension of the C matrix.
///
/// # Safety
///
/// The caller MUST ensure the following invariants to prevent undefined behavior:
/// - All pointers (`a_panel`, `b_panel`, `c_micropanel`) must be valid.
/// - `c_micropanel` must point to a memory region large enough to hold an `mr x nr`
///   submatrix with a column stride of `m`.
/// - `mr <= MR`, `nr <= NR`, `kc <= KC`.
/// - **CRITICAL FLAW IN THIS IMPLEMENTATION:** This kernel unconditionally loads and stores
///   `NR` (8) columns of C. If `nr < 8`, it will read and write out of the bounds of
///   the `c_micropanel`. A production-ready implementation MUST handle the `nr < NR`
///   edge case, for example by using scalar code for the remaining columns or by
///   using masked stores if the SIMD abstraction supports them. The `mr` dimension is
///   handled correctly via masked loads/stores provided by `simdly`.
unsafe fn kernel_8x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    println!("kernel_8x8 mr: {}, nr: {}.", mr, nr);

    // These registers will accumulate the results for an 8x8 block of C.
    // Each register holds one COLUMN of the 8x8 C block.
    // The `F32x8::load` with `mr` correctly handles the case `mr < 8` by masking.
    // NOTE: The loads for c1 through c7 are UNSAFE if `nr` is less than their respective column index.
    let mut c0 = F32x8::load(c_micropanel.add(0), mr);
    let mut c1 = F32x8::load(c_micropanel.add(m), mr);
    let mut c2 = F32x8::load(c_micropanel.add(2 * m), mr);
    let mut c3 = F32x8::load(c_micropanel.add(3 * m), mr);
    let mut c4 = F32x8::load(c_micropanel.add(4 * m), mr);
    let mut c5 = F32x8::load(c_micropanel.add(5 * m), mr);
    let mut c6 = F32x8::load(c_micropanel.add(6 * m), mr);
    let mut c7 = F32x8::load(c_micropanel.add(7 * m), mr);

    let c_cols: Vec<F32x8> = (0..nr)
        .map(|i| F32x8::load(c_micropanel.add(i * m), mr))
        .collect();

    // Loop over the K dimension, `p` from 0 to `kc-1`.
    for p in 0..kc {
        // Get pointers to the p-th column of packed A and p-th row of packed B.
        let a_col = &a_panel.data[p];
        let b_row = &b_panel.data[p];

        let a = F32x8::load_aligned(a_col.as_ptr());
        let b = F32x8::load_aligned(b_row.as_ptr());

        [c0, c1, c2, c3, c4, c5, c6, c7] =
            outer_product(a, b, &mut [c0, c1, c2, c3, c4, c5, c6, c7]);
    }

    // Store the accumulated results back to the C matrix.
    // The `store_at` method correctly handles `mr < 8` by masking.
    // NOTE: The stores for c1 through c7 are UNSAFE if `nr` is less than their respective column index.
    c0.store_at(c_micropanel.add(0));
    c1.store_at(c_micropanel.add(m));
    c2.store_at(c_micropanel.add(2 * m));
    c3.store_at(c_micropanel.add(3 * m));
    c4.store_at(c_micropanel.add(4 * m));
    c5.store_at(c_micropanel.add(5 * m));
    c6.store_at(c_micropanel.add(6 * m));
    c7.store_at(c_micropanel.add(7 * m));
}

unsafe fn kernel_MRxNR(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let c_cols: Vec<F32x8> = (0..nr)
        .map(|i| F32x8::load(c_micropanel.add(i * m), mr))
        .collect();

    for c_col in c_cols {
        println!("c_col: {:?}", c_col.to_vec());
    }
}

// --- Verification and Main ---

/// A simple, unoptimized reference implementation of column-major matrix multiplication.
/// Used for verifying the correctness of the optimized version.
fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                // Access elements in column-major order.
                sum += a[at(row, inner, m)] * b[at(inner, col, k)];
            }
            c[at(row, col, m)] = sum;
        }
    }
}

/// .
/// # Safety
/// .
#[target_feature(enable = "avx2")]
pub unsafe fn outer_product(a: F32x8, b: F32x8, result: &mut [F32x8; 8]) -> [F32x8; 8] {
    // Pre-compute both halves to maximize instruction-level parallelism
    let a_lo = a.permute2f128::<0x00>(a);
    let a_hi = a.permute2f128::<0x11>(a);

    // // Interleave operations to maximize port utilization on modern CPUs
    // // Intel CPUs have multiple execution ports - we want to use them all

    // Group 1: Prepare broadcasts for elements 0,1,4,5 (interleaved)
    let a0 = a_lo.permute::<0x00>(); // Port 5
    let a4 = a_hi.permute::<0x00>(); // Port 5 (parallel)
    let a1 = a_lo.permute::<0x55>(); // Port 5
    let a5 = a_hi.permute::<0x55>(); // Port 5 (parallel)

    // // Group 2: Compute multiplications for rows 0,4,1,5 (interleaved)

    result[0] += a0 * b; // Port 0/1
    result[4] += a4 * b; // Port 0/1 (parallel)
    result[1] += a1 * b; // Port 0/1
    result[5] += a5 * b; // Port 0/1 (parallel)

    // Group 3: Prepare broadcasts for elements 2,3,6,7 (interleaved)
    let a2 = a_lo.permute::<0xAA>(); // Port 5
    let a6 = a_hi.permute::<0xAA>(); // Port 5 (parallel)
    let a3 = a_lo.permute::<0xFF>(); // Port 5
    let a7 = a_hi.permute::<0xFF>(); // Port 5 (parallel)

    // // Group 4: Compute multiplications for rows 2,6,3,7 (interleaved)
    result[2] += a2 * b; // Port 0/1
    result[6] += a6 * b; // Port 0/1 (parallel)
    result[3] += a3 * b; // Port 0/1
    result[7] += a7 * b; // Port 0/1 (parallel)

    *result
}

/// .
/// # Safety
/// .
#[target_feature(enable = "avx2")]
pub unsafe fn outer_product_fma(a: F32x8, b: F32x8, result: &mut [F32x8; 8]) -> [F32x8; 8] {
    // Pre-compute both halves to maximize instruction-level parallelism
    let a_lo = a.permute2f128::<0x00>(a);
    let a_hi = a.permute2f128::<0x11>(a);

    // // Interleave operations to maximize port utilization on modern CPUs
    // // Intel CPUs have multiple execution ports - we want to use them all

    // Group 1: Prepare broadcasts for elements 0,1,4,5 (interleaved)
    let a0 = a_lo.permute::<0x00>(); // Port 5
    let a4 = a_hi.permute::<0x00>(); // Port 5 (parallel)
    let a1 = a_lo.permute::<0x55>(); // Port 5
    let a5 = a_hi.permute::<0x55>(); // Port 5 (parallel)

    // // Group 2: Compute multiplications for rows 0,4,1,5 (interleaved)
    result[0].fmadd(a0, b); // Port 0/1
    result[4].fmadd(a4, b); // Port 0/1 (parallel)
    result[1].fmadd(a1, b); // Port 0/1
    result[5].fmadd(a5, b); // Port 0/1 (

    // Group 3: Prepare broadcasts for elements 2,3,6,7 (interleaved)
    let a2 = a_lo.permute::<0xAA>(); // Port 5
    let a6 = a_hi.permute::<0xAA>(); // Port 5 (parallel)
    let a3 = a_lo.permute::<0xFF>(); // Port 5
    let a7 = a_hi.permute::<0xFF>(); // Port 5 (parallel)

    // Group 4: Compute multiplications for rows 2,6,3,7 (interleaved)
    result[2].fmadd(a2, b);
    result[6].fmadd(a6, b);
    result[3].fmadd(a3, b);
    result[7].fmadd(a7, b);

    *result
}

/// Main function to drive a test case.
fn main() {
    // Define matrix dimensions. Using multiples of 8 helps test the main kernel path.
    let m = 3;
    let k = 3;
    let n = 3;

    // Initialize matrices with simple values for easy verification.
    let a = vec![1.0; m * k];
    let b = vec![1.0; k * n];

    // Allocate zero-initialized output matrices.
    // let mut c_matmul = vec![0.0; m * n];
    // let mut c_matmul = alloc_zeroed_f32_vec(m * n, 32);
    let mut c_matmul = (0..m * n).map(|i| i as f32).collect::<Vec<f32>>();
    // let mut c_naive_matmul = vec![0.0; m * n];

    // Run both the optimized and naive implementations.
    matmul(&a, &b, &mut c_matmul, m, n, k);
    // naive_matmul(&a, &b, &mut c_naive_matmul, m, n, k);

    // // Verify that the results are identical.
    // let mut mismatch_found = false;
    // for i in 0..m * n {
    //     if (c_matmul[i] - c_naive_matmul[i]).abs() > 1e-6 {
    //         let row = i % m;
    //         let col = i / m;
    //         println!(
    //             "Mismatch at ({}, {}): Optimized = {}, Naive = {}",
    //             row, col, c_matmul[i], c_naive_matmul[i]
    //         );
    //         mismatch_found = true;
    //         break;
    //     }
    // }

    // if !mismatch_found {
    //     println!("All values match between optimized and naive matmul implementations.");
    // }
}
