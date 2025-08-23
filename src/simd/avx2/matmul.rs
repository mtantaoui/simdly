//! # High-Performance Matrix Multiplication (GEMM)
//!
//! This module provides a cache-blocked implementation of general matrix-matrix
//! multiplication, `C = A * B`, for single-precision floating-point numbers (`f32`).
//!
//! The implementation follows the BLIS (BLAS-like Library Instantiation Software) algorithm
//! design, which achieves good performance through:
//!
//! 1. **Cache-Aware Blocking**: Matrices are partitioned into blocks (MC×KC, KC×NC, MC×NC)
//!    that fit within CPU cache levels to minimize memory traffic.
//!
//! 2. **Data Packing**: Matrix blocks are copied into contiguous, aligned buffers with
//!    layouts optimized for the microkernel:
//!    - A panels: Column-major layout for efficient vector loads
//!    - B panels: Row-major layout for efficient broadcasting
//!
//! 3. **SIMD Microkernel**: An 8×8 microkernel using AVX2 F32x8 vectors performs the
//!    core computation with:
//!    - Fused multiply-add (FMA) operations
//!    - SIMD shuffle-based broadcasting
//!    - Automatic handling of partial blocks
//!
//! **Matrix Storage**: All matrices use **column-major** order (Fortran-style),
//! where element (i,j) is at index `j * leading_dimension + i`.

use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

use crate::simd::avx2::f32x8::F32x8;
use crate::simd::{SimdMath, SimdShuffle, SimdStore};

// --- Algorithm Constants ---

/// The memory alignment required for AVX2 SIMD instructions. AVX instructions operate
/// on 256-bit (32-byte) registers, so aligning memory to 32-byte boundaries
/// allows for the use of faster, aligned load/store instructions.
const ALIGNMENT: usize = 32;

/// Microkernel row dimension: Number of rows processed simultaneously.
/// Set to 8 to match AVX2 F32x8 vector width (8 × 32-bit floats = 256 bits).
pub const MR: usize = 8;

/// Microkernel column dimension: Number of columns processed simultaneously.
/// Set to 8 to balance register usage and cache efficiency for the 8×8 microkernel.
pub const NR: usize = 8;

/// L1 cache block size for K dimension (inner product length).
/// Chosen so that MR×KC panel of A and KC×NR panel of B fit in L1 cache.
pub const KC: usize = 32;

/// L2 cache block size for M dimension (A rows, C rows).
/// Must be a multiple of MR. Tuned for L2 cache capacity.
pub const MC: usize = 32; // 2 × MR

/// L3 cache block size for N dimension (B columns, C columns).
/// Must be a multiple of NR. Tuned for L3 cache capacity.
pub const NC: usize = 32; // 2 × NR

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

/// Utility function to print a matrix stored in row-major format (for debugging).
pub fn display_matrix_row_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[i * ld + j]); // Row-major indexing
        }
        println!()
    }
    println!("---");
}

// --- Packed Data Structures for Matrix B ---

/// A packed panel of matrix B with dimensions KC×NR.
///
/// Data is stored in **row-major** format: `data[k][j]` contains B(k,j) where:
/// - k ranges from 0 to KC-1 (rows from the KC×NR block of B)
/// - j ranges from 0 to NR-1 (columns within this NR-wide panel)
///
/// This layout enables the microkernel to:
/// 1. Load a full row `data[k]` to get B(k, 0..NR-1)
/// 2. Broadcast individual elements B(k,j) efficiently
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for AVX2 operations.
#[repr(C, align(32))]
pub struct BPanel<const KC: usize, const NR: usize> {
    pub data: [[f32; NR]; KC],
}

/// A heap-allocated block containing multiple B panels.
///
/// Manages memory for ceil(nc/NR) panels of KC×NR elements each, where nc is the
/// number of columns from the original B matrix block being packed.
///
/// Memory layout: [Panel₀][Panel₁]...[Panelₙ] with 32-byte alignment.
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
    /// Allocates zero-initialized, aligned memory for packing nc columns.
    ///
    /// Creates ceil(nc/NR) panels to hold all nc columns. Panels beyond the first
    /// may be partially filled if nc is not a multiple of NR.
    ///
    /// # Arguments
    /// * `nc` - Number of columns from original B matrix to pack
    ///
    /// # Returns
    /// New BBlock or allocation error
    #[inline(always)]
    pub fn new(nc: usize) -> Result<Self, Layout> {
        // Calculate panels needed: ceil(nc / NR)
        let num_panels = nc.div_ceil(NR);

        // Define the memory layout for an array of `BPanel`s.
        let layout = Layout::array::<BPanel<KC, NR>>(num_panels).unwrap();

        // Ensure the layout meets our minimum alignment requirement.
        let layout = layout.align_to(ALIGNMENT).unwrap();

        let ptr = unsafe {
            // Zero-initialize since partial panels need zero-padding
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
    #[inline(always)]
    pub fn as_panels(&self) -> &[BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `BPanel`s.
    #[inline(always)]
    pub fn as_panels_mut(&mut self) -> &mut [BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// The `Drop` implementation ensures that the heap-allocated memory is safely
/// deallocated when a `BBlock` goes out of scope.
impl<const KC: usize, const NR: usize> Drop for BBlock<KC, NR> {
    #[inline(always)]
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
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const KC: usize, const NR: usize> IndexMut<usize> for BBlock<KC, NR> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Packed Data Structures for Matrix A ---

/// A packed panel of matrix A with dimensions MR×KC.
///
/// Data is stored to optimize microkernel access: `data[k]` contains the k-th column
/// of an MR×KC block from matrix A, i.e., A(0..MR-1, k).
///
/// This layout enables the microkernel to:
/// 1. Load `data[k]` as a vector: A(0..MR-1, k)
/// 2. Process entire columns efficiently with SIMD
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for AVX2 loads.
#[repr(C, align(32))]
pub struct APanel<const MR: usize, const KC: usize> {
    pub data: [[f32; MR]; KC],
}

/// A heap-allocated block containing multiple A panels.
///
/// Manages memory for ceil(mc/MR) panels of MR×KC elements each, where mc is the
/// number of rows from the original A matrix block being packed.
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
    /// Allocates zero-initialized, aligned memory for packing mc rows.
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const MR: usize, const KC: usize> IndexMut<usize> for ABlock<MR, KC> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Core Matrix Multiplication Logic ---

/// Computes C += A × B using the BLIS algorithm with cache blocking and data packing.
///
/// Implements the five nested loops of BLIS:
/// - jc: Iterate over N in blocks of NC (L3 cache blocking)
/// - pc: Iterate over K in blocks of KC (reuse packed B)
/// - ic: Iterate over M in blocks of MC (L2 cache blocking)
/// - jr: Iterate over packed B panels (NR-wide)
/// - ir: Iterate over packed A panels (MR-wide)
///
/// # Arguments
/// * `a` - Matrix A (m×k) in column-major order
/// * `b` - Matrix B (k×n) in column-major order  
/// * `c` - Matrix C (m×n) in column-major order (input/output)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Inner dimension (columns of A, rows of B)
///
/// # Panics
/// Panics if matrix dimensions don't match slice lengths.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    matmul_with_params(a, b, c, m, n, k, MC, NC);
}

/// Column-major matrix multiplication using optimized BLIS algorithm with runtime MC/NC parameters.
///
/// This function allows runtime specification of MC and NC cache blocking parameters for performance tuning.
/// KC remains a compile-time constant to avoid complex const generic refactoring.
///
/// # Arguments
/// * `a` - Left operand matrix A in column-major format
/// * `b` - Right operand matrix B in column-major format  
/// * `c` - Output matrix C in column-major format (will be overwritten)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B (shared dimension)
/// * `mc` - Cache blocking parameter: rows of A block (should be multiple of MR=8)
/// * `nc` - Cache blocking parameter: columns of B block (should be multiple of NR=8)  
///
/// # Panics
/// Panics if matrix dimensions are incompatible or if any matrix slice is too small.
pub fn matmul_with_params(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    nc: usize,
) {
    // Handle edge cases
    if m == 0 || n == 0 || k == 0 {
        return; // Nothing to compute
    }

    // Verify input dimensions
    assert_eq!(a.len(), m * k, "Matrix A has incorrect dimensions");
    assert_eq!(b.len(), k * n, "Matrix B has incorrect dimensions");
    assert_eq!(c.len(), m * n, "Matrix C has incorrect dimensions");
    // jc loop: Process N dimension in nc-wide blocks (L3 cache optimization)
    for jc in (0..n).step_by(nc) {
        let nc_actual = min(nc, n - jc);

        // pc loop: Process K dimension in KC-wide blocks
        // Placed outside ic loop to reuse packed B across multiple A blocks
        for pc in (0..k).step_by(KC) {
            let kc_actual = min(KC, k - pc);

            // Pack B block: B(pc:pc+kc_actual-1, jc:jc+nc_actual-1) → row-major panels
            let b_block = pack_b::<KC, NR>(b, nc_actual, kc_actual, k, pc, jc);

            // ic loop: Process M dimension in mc-wide blocks (L2 cache optimization)
            for ic in (0..m).step_by(mc) {
                let mc_actual = min(mc, m - ic);

                // Pack A block: A(ic:ic+mc_actual-1, pc:pc+kc_actual-1) → column-major panels
                let a_block = pack_a::<MR, KC>(a, mc_actual, kc_actual, m, ic, pc);

                // jr and ir loops: Process packed panels with MR×NR microkernel
                for jr in 0..b_block.as_panels().len() {
                    let b_panel = &b_block[jr];
                    let nr = min(NR, nc_actual - jr * NR);

                    for ir in 0..a_block.as_panels().len() {
                        let a_panel = &a_block[ir];
                        let mr = min(MR, mc_actual - ir * MR);

                        // Compute C(ic+ir*MR:ic+ir*MR+mr-1, jc+jr*NR:jc+jr*NR+nr-1)
                        let c_row = ic + ir * MR;
                        let c_col = jc + jr * NR;
                        let c_micropanel_start_idx = at(c_row, c_col, m);
                        let c_micropanel = &mut c[c_micropanel_start_idx..];

                        // Execute MR×NR microkernel with AVX2 optimization
                        unsafe {
                            kernel_MRxNR(
                                a_panel,
                                b_panel,
                                c_micropanel.as_mut_ptr(),
                                mr,
                                nr,
                                kc_actual,
                                m,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Packs an mc×kc block of matrix A into cache-friendly panels.
///
/// Extracts A(ic:ic+mc-1, pc:pc+kc-1) and reorganizes it into MR-wide row panels,
/// where each panel stores KC columns in a layout optimized for microkernel access.
///
/// # Arguments
/// * `a` - Source matrix A in column-major order
/// * `mc`, `kc` - Block dimensions to pack  
/// * `m` - Leading dimension (number of rows) of matrix A
/// * `ic`, `pc` - Top-left coordinates of block in A
#[inline(always)]
pub fn pack_a<const MR: usize, const KC: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,
    ic: usize,
    pc: usize,
) -> ABlock<MR, KC> {
    let mut packed_block = ABlock::<MR, KC>::new(mc).expect("Memory allocation failed for ABlock");

    // Process mc rows in groups of MR (microkernel row dimension)
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let mr_in_panel = min(MR, mc - i_panel_start); // Handle partial panels

        // Pack all KC columns of this row panel
        for p_col in 0..kc {
            // Copy column from A(ic+i_panel_start:ic+i_panel_start+mr_in_panel-1, pc+p_col)
            let src_col = pc + p_col;
            let src_row_start = ic + i_panel_start;
            let src_start = at(src_row_start, src_col, m);
            let src_slice = &a[src_start..src_start + mr_in_panel];
            let dest_slice = &mut dest_panel.data[p_col][0..mr_in_panel];
            dest_slice.copy_from_slice(src_slice);
        }
    }

    packed_block
}

/// Packs a kc×nc block of matrix B into cache-friendly panels.
///
/// Extracts B(pc:pc+kc-1, jc:jc+nc-1) and reorganizes it into NR-wide column panels,
/// where each panel stores KC rows in row-major format for efficient broadcasting.
///
/// # Arguments  
/// * `b` - Source matrix B in column-major order
/// * `nc`, `kc` - Block dimensions to pack
/// * `k` - Leading dimension (number of rows) of matrix B
/// * `pc`, `jc` - Top-left coordinates of block in B
#[inline(always)]
pub fn pack_b<const KC: usize, const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,
    pc: usize,
    jc: usize,
) -> BBlock<KC, NR> {
    let mut packed_block = BBlock::<KC, NR>::new(nc).expect("Memory allocation failed for BBlock");

    // Process nc columns in groups of NR (microkernel column dimension)
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let nr_in_panel = min(NR, nc - j_panel_start); // Handle partial panels

        // Pack KC rows of this column panel in row-major order
        for p_row in 0..kc {
            // Pack row pc+p_row across columns jc+j_panel_start:jc+j_panel_start+nr_in_panel-1
            let src_row = pc + p_row;
            for j_col_in_panel in 0..nr_in_panel {
                let src_col = jc + j_panel_start + j_col_in_panel;
                let src_idx = at(src_row, src_col, k); // B(src_row, src_col)
                dest_panel.data[p_row][j_col_in_panel] = b[src_idx];
            }
        }
    }

    packed_block
}

/// AVX2-optimized microkernel computing C += A × B for MR×NR blocks.
///
/// Uses F32x8 vectors and SIMD shuffle operations for high performance:
/// - Loads A columns as F32x8 vectors
/// - Uses permute operations for efficient B element broadcasting  
/// - Applies FMA operations for C += A × B computation
/// - Handles partial blocks automatically via F32x8 size tracking
///
/// # Arguments
/// * `a_panel` - Packed A panel (MR×KC)
/// * `b_panel` - Packed B panel (KC×NR)  
/// * `c_micropanel` - Pointer to C block (MR×NR)
/// * `mr`, `nr` - Actual dimensions (≤ MR, NR)
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of C
///
/// # Safety
/// Caller must ensure c_micropanel has at least mr×nr elements accessible
/// in column-major order with leading dimension m.
#[allow(non_snake_case)]
#[inline(always)]
unsafe fn kernel_MRxNR(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    // Optimized 8xNR microkernel (MR=8, NR≤8)
    // Load existing C values for accumulation (C += A*B)
    let mut c_cols = [F32x8::zeros(); NR];

    for j in 0..nr {
        c_cols[j] = F32x8::from(std::slice::from_raw_parts(c_micropanel.add(j * m), mr));
    }

    // Unroll KC loop completely for maximum performance
    for k in 0..kc {
        // Load A column k: 8 elements from A(0..7, k)
        let a_vec = F32x8::from(&a_panel.data[k][0..mr]);

        // Load B row k: B(k, 0..nr-1) efficiently
        let b_data = &b_panel.data[k];

        // 8xNR FMA operations: C[0..7, j] += A[0..7, k] * B[k, j]
        // Use optimized SIMD shuffle-based broadcasting for maximum performance
        match nr {
            8 => {
                // Load B values into SIMD vector for efficient shuffling
                let b_vec = F32x8::from(b_data.as_slice());

                // Use SIMD shuffle operations for efficient broadcasting
                let b_lower = b_vec.permute2f128::<0x00>(); // [b0,b1,b2,b3, b0,b1,b2,b3]
                let b_upper = b_vec.permute2f128::<0x11>(); // [b4,b5,b6,b7, b4,b5,b6,b7]

                // Broadcast individual elements using permute
                let b0_bc = b_lower.permute::<0x00>(); // [b0, b0, b0, b0, b0, b0, b0, b0]
                let b1_bc = b_lower.permute::<0x55>(); // [b1, b1, b1, b1, b1, b1, b1, b1]
                let b2_bc = b_lower.permute::<0xAA>(); // [b2, b2, b2, b2, b2, b2, b2, b2]
                let b3_bc = b_lower.permute::<0xFF>(); // [b3, b3, b3, b3, b3, b3, b3, b3]
                let b4_bc = b_upper.permute::<0x00>(); // [b4, b4, b4, b4, b4, b4, b4, b4]
                let b5_bc = b_upper.permute::<0x55>(); // [b5, b5, b5, b5, b5, b5, b5, b5]
                let b6_bc = b_upper.permute::<0xAA>(); // [b6, b6, b6, b6, b6, b6, b6, b6]
                let b7_bc = b_upper.permute::<0xFF>(); // [b7, b7, b7, b7, b7, b7, b7, b7]

                // Optimized 8-way unrolled FMA operations using shuffle-based broadcasts
                c_cols[0] = c_cols[0].fma(a_vec, b0_bc);
                c_cols[1] = c_cols[1].fma(a_vec, b1_bc);
                c_cols[2] = c_cols[2].fma(a_vec, b2_bc);
                c_cols[3] = c_cols[3].fma(a_vec, b3_bc);
                c_cols[4] = c_cols[4].fma(a_vec, b4_bc);
                c_cols[5] = c_cols[5].fma(a_vec, b5_bc);
                c_cols[6] = c_cols[6].fma(a_vec, b6_bc);
                c_cols[7] = c_cols[7].fma(a_vec, b7_bc);
            }
            7 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();
                let b_upper = b_vec.permute2f128::<0x11>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
                c_cols[2] = c_cols[2].fma(a_vec, b_lower.permute::<0xAA>());
                c_cols[3] = c_cols[3].fma(a_vec, b_lower.permute::<0xFF>());
                c_cols[4] = c_cols[4].fma(a_vec, b_upper.permute::<0x00>());
                c_cols[5] = c_cols[5].fma(a_vec, b_upper.permute::<0x55>());
                c_cols[6] = c_cols[6].fma(a_vec, b_upper.permute::<0xAA>());
            }
            6 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();
                let b_upper = b_vec.permute2f128::<0x11>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
                c_cols[2] = c_cols[2].fma(a_vec, b_lower.permute::<0xAA>());
                c_cols[3] = c_cols[3].fma(a_vec, b_lower.permute::<0xFF>());
                c_cols[4] = c_cols[4].fma(a_vec, b_upper.permute::<0x00>());
                c_cols[5] = c_cols[5].fma(a_vec, b_upper.permute::<0x55>());
            }
            5 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();
                let b_upper = b_vec.permute2f128::<0x11>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
                c_cols[2] = c_cols[2].fma(a_vec, b_lower.permute::<0xAA>());
                c_cols[3] = c_cols[3].fma(a_vec, b_lower.permute::<0xFF>());
                c_cols[4] = c_cols[4].fma(a_vec, b_upper.permute::<0x00>());
            }
            4 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
                c_cols[2] = c_cols[2].fma(a_vec, b_lower.permute::<0xAA>());
                c_cols[3] = c_cols[3].fma(a_vec, b_lower.permute::<0xFF>());
            }
            3 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
                c_cols[2] = c_cols[2].fma(a_vec, b_lower.permute::<0xAA>());
            }
            2 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
                c_cols[1] = c_cols[1].fma(a_vec, b_lower.permute::<0x55>());
            }
            1 => {
                let b_vec = F32x8::from(b_data.as_slice());
                let b_lower = b_vec.permute2f128::<0x00>();

                c_cols[0] = c_cols[0].fma(a_vec, b_lower.permute::<0x00>());
            }
            _ => {}
        }
    }

    // Store results back to C matrix
    match nr {
        8 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
            c_cols[3].store_at(c_micropanel.add(3 * m));
            c_cols[4].store_at(c_micropanel.add(4 * m));
            c_cols[5].store_at(c_micropanel.add(5 * m));
            c_cols[6].store_at(c_micropanel.add(6 * m));
            c_cols[7].store_at(c_micropanel.add(7 * m));
        }
        7 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
            c_cols[3].store_at(c_micropanel.add(3 * m));
            c_cols[4].store_at(c_micropanel.add(4 * m));
            c_cols[5].store_at(c_micropanel.add(5 * m));
            c_cols[6].store_at(c_micropanel.add(6 * m));
        }
        6 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
            c_cols[3].store_at(c_micropanel.add(3 * m));
            c_cols[4].store_at(c_micropanel.add(4 * m));
            c_cols[5].store_at(c_micropanel.add(5 * m));
        }
        5 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
            c_cols[3].store_at(c_micropanel.add(3 * m));
            c_cols[4].store_at(c_micropanel.add(4 * m));
        }
        4 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
            c_cols[3].store_at(c_micropanel.add(3 * m));
        }
        3 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
            c_cols[2].store_at(c_micropanel.add(2 * m));
        }
        2 => {
            c_cols[0].store_at(c_micropanel);
            c_cols[1].store_at(c_micropanel.add(m));
        }
        1 => {
            c_cols[0].store_at(c_micropanel);
        }
        _ => {}
    }
}

// --- Reference Implementation for Testing ---

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(mnk) matrix multiplication for correctness verification.
    /// Computes C += A × B using the standard triple-nested loop algorithm.
    fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        // Standard IJK algorithm: C[i,j] += ∑(A[i,k] * B[k,j])
        for j in 0..n {
            // For each column of C
            for i in 0..m {
                // For each row of C
                let mut sum = 0.0;
                for l in 0..k {
                    // Inner product
                    sum += a[at(i, l, m)] * b[at(l, j, k)];
                }
                c[at(i, j, m)] += sum; // Accumulate into C[i,j]
            }
        }
    }

    /// Creates test matrix with values (row+1) + (col+1)*0.1 for easy verification.
    /// Examples: A[0,0]=1.1, A[1,0]=2.1, A[0,1]=1.2, A[1,1]=2.2
    fn create_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let idx = at(row, col, rows);
                matrix[idx] = (row + 1) as f32 + (col + 1) as f32 * 0.1;
            }
        }
        matrix
    }

    /// Creates identity matrix: I[i,j] = 1 if i==j, 0 otherwise.
    fn create_identity_matrix(size: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; size * size];
        for i in 0..size {
            let idx = at(i, i, size);
            matrix[idx] = 1.0;
        }
        matrix
    }

    /// Verifies the at() function correctly implements column-major indexing.
    #[test]
    fn test_column_major_indexing() {
        // Column-major storage: [col0, col1] = [a,b,c, d,e,f]
        // Matrix layout:  | a d |
        //                 | b e |
        //                 | c f |
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 3; // rows
        let _n = 2; // cols

        assert_eq!(matrix[at(0, 0, m)], 1.0); // top-left
        assert_eq!(matrix[at(1, 0, m)], 2.0); // middle-left
        assert_eq!(matrix[at(2, 0, m)], 3.0); // bottom-left
        assert_eq!(matrix[at(0, 1, m)], 4.0); // top-right
        assert_eq!(matrix[at(1, 1, m)], 5.0); // middle-right
        assert_eq!(matrix[at(2, 1, m)], 6.0); // bottom-right
    }

    /// Test matrix multiplication with identity matrix
    #[test]
    fn test_matmul_identity() {
        let size = 4;
        let a = create_test_matrix(size, size);
        let identity = create_identity_matrix(size);
        let mut c_optimized = vec![0.0; size * size];
        let mut c_naive = vec![0.0; size * size];

        // A * I = A
        matmul(&a, &identity, &mut c_optimized, size, size, size);
        naive_matmul(&a, &identity, &mut c_naive, size, size, size);

        // Verify both implementations match
        for i in 0..size * size {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "Mismatch at index {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }

        // Verify A * I = A
        for i in 0..size * size {
            assert!(
                (c_optimized[i] - a[i]).abs() < 1e-6,
                "A*I != A at index {}: result={}, original={}",
                i,
                c_optimized[i],
                a[i]
            );
        }
    }

    /// Tests 2×2 multiplication with manually computed expected result.
    #[test]
    fn test_matmul_2x2() {
        // A = | 1 3 |   B = | 5 7 |   C = | 1*5+3*6  1*7+3*8 | = | 23 31 |
        //     | 2 4 |       | 6 8 |       | 2*5+4*6  2*7+4*8 |   | 34 46 |

        // Column-major: A=[1,2, 3,4] (col0=[1,2], col1=[3,4])
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_optimized = vec![0.0; 4];
        let mut c_naive = vec![0.0; 4];
        let m = 2;
        let n = 2;
        let k = 2;

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        // Expected C in column-major: [23,34, 31,46] (col0=[23,34], col1=[31,46])
        let expected = vec![23.0, 34.0, 31.0, 46.0];

        for i in 0..4 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "Optimized mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_naive[i] - expected[i]).abs() < 1e-6,
                "Naive mismatch at {}: got {}, expected {}",
                i,
                c_naive[i],
                expected[i]
            );
        }
    }

    /// Test with 3x3 matrices
    #[test]
    fn test_matmul_3x3() {
        let m = 3;
        let n = 3;
        let k = 3;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "3x3 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test non-square matrices
    #[test]
    fn test_matmul_non_square() {
        // Test 2x3 * 3x4 = 2x4
        let m = 2;
        let k = 3;
        let n = 4;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "2x3*3x4 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests matrices that are multiples of blocking parameters (MC,NC,KC).
    #[test]
    fn test_matmul_large_blocks() {
        let m = 16;
        let n = 16;
        let k = 16; // Multiples of MR, NR, KC
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-3,
                "16x16 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests dimensions that don't align with blocking parameters.
    #[test]
    fn test_matmul_odd_dimensions() {
        let m = 7;
        let n = 5;
        let k = 9; // Not multiples of MR=8, NR=8, KC=16
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "7x9*9x5 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test with zeros
    #[test]
    fn test_matmul_zeros() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a = vec![0.0; m * k];
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "Zero matrix mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
            assert!(
                c_optimized[i].abs() < 1e-6,
                "Result should be zero at {}: {}",
                i,
                c_optimized[i]
            );
        }
    }

    /// Verifies accumulation behavior by pre-initializing C with non-zero values.
    #[test]
    fn test_matmul_accumulation() {
        let m = 3;
        let n = 3;
        let k = 3;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![1.0; m * n]; // Pre-filled with ones
        let mut c_naive = vec![1.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "Accumulation mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test 1x1 matrix multiplication
    #[test]
    fn test_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c_optimized = vec![0.0; 1];
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 1);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 1);

        assert!(
            (c_optimized[0] - 12.0).abs() < 1e-6,
            "1x1 result should be 12.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x1 optimized vs naive mismatch"
        );
    }

    /// Tests matrices smaller than microkernel dimensions.
    #[test]
    fn test_matmul_small_matrices() {
        // 1×2 × 2×1 → 1×1
        let a = vec![1.0, 2.0]; // 1x2 in column-major: [1, 2]
        let b = vec![3.0, 4.0]; // 2x1 in column-major: [3, 4]
        let mut c_optimized = vec![0.0; 1]; // 1x1 result
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 2);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 2);

        // Expected: 1*3 + 2*4 = 11
        assert!(
            (c_optimized[0] - 11.0).abs() < 1e-6,
            "1x2*2x1 result should be 11.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x2*2x1 optimized vs naive mismatch"
        );

        // 2×1 × 1×2 → 2×2
        let a = vec![1.0, 2.0]; // 2x1 in column-major: [1, 2]
        let b = vec![3.0, 4.0]; // 1x2 in column-major: [3, 4]
        let mut c_optimized = vec![0.0; 4]; // 2x2 result
        let mut c_naive = vec![0.0; 4];

        matmul(&a, &b, &mut c_optimized, 2, 2, 1);
        naive_matmul(&a, &b, &mut c_naive, 2, 2, 1);

        // Expected: [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
        // In column-major: [3, 6, 4, 8]
        let expected = vec![3.0, 6.0, 4.0, 8.0];
        for i in 0..4 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "2x1*1x2 mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "2x1*1x2 optimized vs naive mismatch at {}",
                i
            );
        }
    }

    /// Test empty matrices
    #[test]
    fn test_matmul_empty() {
        // Test 0x0 matrices
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c_optimized: Vec<f32> = vec![];
        let mut c_naive: Vec<f32> = vec![];

        matmul(&a, &b, &mut c_optimized, 0, 0, 0);
        naive_matmul(&a, &b, &mut c_naive, 0, 0, 0);

        assert!(c_optimized.is_empty(), "0x0 result should be empty");
        assert_eq!(c_optimized, c_naive, "0x0 optimized vs naive should match");

        // Test matrices with one dimension being 0
        let a = vec![1.0, 2.0]; // 2x1
        let b: Vec<f32> = vec![]; // 1x0
        let mut c_optimized: Vec<f32> = vec![]; // 2x0
        let mut c_naive: Vec<f32> = vec![];

        matmul(&a, &b, &mut c_optimized, 2, 0, 1);
        naive_matmul(&a, &b, &mut c_naive, 2, 0, 1);

        assert!(c_optimized.is_empty(), "2x0 result should be empty");
        assert_eq!(c_optimized, c_naive, "2x0 optimized vs naive should match");
    }

    /// Tests very tall/thin matrices that stress the blocking algorithm.
    #[test]
    fn test_matmul_extreme_aspect_ratios() {
        // Tall matrix: 8×1 × 1×1 → 8×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8x1
        let b = vec![2.0]; // 1x1
        let mut c_optimized = vec![0.0; 8]; // 8x1
        let mut c_naive = vec![0.0; 8];

        matmul(&a, &b, &mut c_optimized, 8, 1, 1);
        naive_matmul(&a, &b, &mut c_naive, 8, 1, 1);

        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        for i in 0..8 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "8x1*1x1 mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "8x1*1x1 optimized vs naive mismatch at {}",
                i
            );
        }

        // Wide matrix: 1×8 × 8×1 → 1×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x8
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 8x1
        let mut c_optimized = vec![0.0; 1]; // 1x1
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 8);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 8);

        // Expected: sum of 1+2+3+4+5+6+7+8 = 36
        assert!(
            (c_optimized[0] - 36.0).abs() < 1e-6,
            "1x8*8x1 result should be 36.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x8*8x1 optimized vs naive mismatch"
        );
    }

    /// Tests numerical stability with very small and large values.
    #[test]
    fn test_matmul_special_values() {
        // Test with very small numbers
        let a = vec![1e-10, 2e-10];
        let b = vec![3e10, 4e10];
        let mut c_optimized = vec![0.0; 1];
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 2);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 2);

        // Expected: 1e-10 * 3e10 + 2e-10 * 4e10 = 3 + 8 = 11
        assert!(
            (c_optimized[0] - 11.0).abs() < 1e-6,
            "Small numbers result should be 11.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "Small numbers optimized vs naive mismatch"
        );
    }

    /// Informal performance comparison between optimized and naive versions.
    #[test]
    fn test_matmul_performance() {
        let m = 32;
        let n = 32;
        let k = 32;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        let start = std::time::Instant::now();
        matmul(&a, &b, &mut c_optimized, m, n, k);
        let optimized_time = start.elapsed();

        let start = std::time::Instant::now();
        naive_matmul(&a, &b, &mut c_naive, m, n, k);
        let naive_time = start.elapsed();

        println!(
            "32x32 matmul - Optimized: {:?}, Naive: {:?}",
            optimized_time, naive_time
        );

        // Verify correctness
        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-2,
                "Performance test mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests matrices smaller than MR×NR to verify edge case handling.
    #[test]
    fn test_matmul_very_small_matrices() {
        let test_cases = vec![
            (1, 1, 1, "1x1x1"),
            (2, 2, 2, "2x2x2"),
            (3, 4, 2, "3x2x4"),
            (5, 3, 7, "5x7x3"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                assert!(
                    (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                    "{} mismatch at {}: optimized={}, naive={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i]
                );
            }
        }
    }

    /// Tests matrices near the microkernel dimensions MR=8, NR=8.
    #[test]
    fn test_matmul_kernel_sized_matrices() {
        let test_cases = vec![
            (8, 8, 8, "8x8x8 exact"),
            (16, 16, 16, "16x16x16 double"),
            (7, 9, 8, "7x8x9 near"),
            (15, 17, 13, "15x13x17 odd"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                assert!(
                    (c_optimized[i] - c_naive[i]).abs() < 1e-3,
                    "{} mismatch at {}: optimized={}, naive={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i]
                );
            }
        }
    }

    /// Tests performance on larger matrices where cache blocking matters.
    #[test]
    fn test_matmul_large_matrices() {
        let test_cases = vec![
            (64, 64, 64, "64x64x64"),
            (128, 128, 128, "128x128x128"),
            (100, 75, 50, "100x50x75 non-square"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            let start = std::time::Instant::now();
            matmul(&a, &b, &mut c_optimized, m, n, k);
            let optimized_time = start.elapsed();

            let start = std::time::Instant::now();
            naive_matmul(&a, &b, &mut c_naive, m, n, k);
            let naive_time = start.elapsed();

            let speedup = naive_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
            println!(
                "{} - Optimized: {:?}, Naive: {:?}, Speedup: {:.2}x",
                name, optimized_time, naive_time, speedup
            );

            // Use relative error tolerance for large values (avoids precision issues)
            for i in 0..(m * n) {
                let diff = (c_optimized[i] - c_naive[i]).abs();
                let max_val = c_optimized[i].abs().max(c_naive[i].abs());
                let relative_error = if max_val > 1e-6 { diff / max_val } else { diff };

                assert!(
                    relative_error < 1e-5,
                    "{} mismatch at {}: optimized={}, naive={}, rel_err={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i],
                    relative_error
                );
            }
        }
    }
}
