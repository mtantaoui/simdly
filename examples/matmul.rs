use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdMath, SimdShuffle, SimdStore};

pub fn display_matrix_row_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[i * ld + j]); // Row-major indexing
        }
        println!()
    }
    println!("---");
}

pub fn display_matrix_column_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
    println!("---");
}

// --- Algorithm Constants ---

/// The memory alignment required for AVX2 SIMD instructions. AVX instructions operate
/// on 256-bit (32-byte) registers, so aligning memory to 32-byte boundaries
/// allows for the use of faster, aligned load/store instructions.
const ALIGNMENT: usize = 32;

const LANE_COUNT: usize = 8;

/// Microkernel row dimension: Number of rows processed simultaneously.
/// Set to 8 to match AVX2 F32x8 vector width (8 × 32-bit floats = 256 bits).
pub const MR: usize = 8;

/// Microkernel column dimension: Number of columns processed simultaneously.
/// Set to 8 to balance register usage and cache efficiency for the 8×8 microkernel.
pub const NR: usize = 8;

/// L1 cache block size for K dimension (inner product length).
/// Chosen so that MR×KC panel of A and KC×NR panel of B fit in L1 cache.
pub const KC: usize = 16;

// /// L2 cache block size for M dimension (A rows, C rows).
// /// Must be a multiple of MR. Tuned for L2 cache capacity.
// pub const MC: usize = 32; // 2 × MR

// /// L3 cache block size for N dimension (B columns, C columns).
// /// Must be a multiple of NR. Tuned for L3 cache capacity.
// pub const NC: usize = 32; // 2 × NR

// --- Helper Functions ---

/// Calculates the 1D index for a 2D element in a column-major matrix.
///
/// # Arguments
/// * `i` - Row index.
/// * `j` - Column index.
/// * `ld` - Leading dimension (number of rows in the matrix).
#[time_graph::instrument]
fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
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
struct BPanel<const KC: usize, const NR: usize> {
    pub data: [[f32; NR]; KC],
}

/// A heap-allocated block containing multiple B panels.
///
/// Manages memory for ceil(nc/NR) panels of KC×NR elements each, where nc is the
/// number of columns from the original B matrix block being packed.
///
/// Memory layout: [Panel₀][Panel₁]...[Panelₙ] with 32-byte alignment.
struct BBlock<const KC: usize, const NR: usize> {
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
    #[time_graph::instrument]
    fn new(nc: usize) -> Result<Self, Layout> {
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
    #[time_graph::instrument]
    fn as_panels(&self) -> &[BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `BPanel`s.
    #[time_graph::instrument]
    fn as_panels_mut(&mut self) -> &mut [BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// The `Drop` implementation ensures that the heap-allocated memory is safely
/// deallocated when a `BBlock` goes out of scope.
impl<const KC: usize, const NR: usize> Drop for BBlock<KC, NR> {
    #[time_graph::instrument]
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
    #[time_graph::instrument]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const KC: usize, const NR: usize> IndexMut<usize> for BBlock<KC, NR> {
    #[time_graph::instrument]
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
struct APanel<const MR: usize, const KC: usize> {
    pub data: [[f32; MR]; KC],
}

/// A heap-allocated block containing multiple A panels.
///
/// Manages memory for ceil(mc/MR) panels of MR×KC elements each, where mc is the
/// number of rows from the original A matrix block being packed.
struct ABlock<const MR: usize, const KC: usize> {
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
    #[time_graph::instrument]
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
    #[time_graph::instrument]
    fn as_panels(&self) -> &[APanel<MR, KC>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `APanel`s.
    #[time_graph::instrument]
    fn as_panels_mut(&mut self) -> &mut [APanel<MR, KC>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// Safe deallocation for `ABlock`'s memory.
impl<const MR: usize, const KC: usize> Drop for ABlock<MR, KC> {
    #[time_graph::instrument]
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

    #[time_graph::instrument]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const MR: usize, const KC: usize> IndexMut<usize> for ABlock<MR, KC> {
    #[time_graph::instrument]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Core Matrix Multiplication Logic ---

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
#[time_graph::instrument]
fn matmul_with_params(
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
            let kc = min(KC, k - pc);

            // Pack B block: B(pc:pc+kc_actual-1, jc:jc+nc_actual-1) → row-major panels
            let b_block = pack_b::<KC, NR>(b, nc_actual, kc, k, pc, jc);

            // ic loop: Process M dimension in mc-wide blocks (L2 cache optimization)
            for ic in (0..m).step_by(mc) {
                let mc_actual = min(mc, m - ic);

                // Pack A block: A(ic:ic+mc_actual-1, pc:pc+kc_actual-1) → column-major panels
                let a_block = pack_a::<MR, KC>(a, mc_actual, kc, m, ic, pc);

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

                        unsafe {
                            kernel(a_panel, b_panel, c_micropanel.as_mut_ptr(), mr, nr, kc, m);
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
#[time_graph::instrument]
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
#[time_graph::instrument]
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

fn outer(a_micropanel: F32x8, b_micropanel: F32x8, result: &mut [F32x8; 8]) -> [F32x8; 8] {
    // Duplicate 128-bit lanes to prepare for element broadcasting
    // permute2f128 with mask 0x00: [a0,a1,a2,a3, a0,a1,a2,a3]
    // permute2f128 with mask 0x11: [a4,a5,a6,a7, a4,a5,a6,a7]
    let a_lower_lane = a_micropanel.permute2f128::<0x00>();
    let a_upper_lane = a_micropanel.permute2f128::<0x11>();

    // First batch: broadcast elements 0 and 4, then 1 and 5
    // Interleave operations to maximize port utilization on modern CPUs
    let a0_broadcast = a_lower_lane.permute::<0x00>(); // [a0, a0, a0, a0, a0, a0, a0, a0]
    let a4_broadcast = a_upper_lane.permute::<0x00>(); // [a4, a4, a4, a4, a4, a4, a4, a4]
    let a1_broadcast = a_lower_lane.permute::<0x55>(); // [a1, a1, a1, a1, a1, a1, a1, a1]
    let a5_broadcast = a_upper_lane.permute::<0x55>(); // [a5, a5, a5, a5, a5, a5, a5, a5]

    // Create size-matched broadcast vectors once for all operations
    let a0_sized = a0_broadcast;
    let a4_sized = a4_broadcast;
    let a1_sized = a1_broadcast;
    let a5_sized = a5_broadcast;

    result[0] = result[0].fma(a0_sized, b_micropanel);
    result[4] = result[4].fma(a4_sized, b_micropanel);
    result[1] = result[1].fma(a1_sized, b_micropanel);
    result[5] = result[5].fma(a5_sized, b_micropanel);

    // Second batch: broadcast elements 2 and 6, then 3 and 7
    let a2_broadcast = a_lower_lane.permute::<0xAA>(); // [a2, a2, a2, a2, a2, a2, a2, a2]
    let a6_broadcast = a_upper_lane.permute::<0xAA>(); // [a6, a6, a6, a6, a6, a6, a6, a6]
    let a3_broadcast = a_lower_lane.permute::<0xFF>(); // [a3, a3, a3, a3, a3, a3, a3, a3]
    let a7_broadcast = a_upper_lane.permute::<0xFF>(); // [a7, a7, a7, a7, a7, a7, a7, a7]

    let a2_sized = a2_broadcast;
    let a6_sized = a6_broadcast;
    let a3_sized = a3_broadcast;
    let a7_sized = a7_broadcast;

    result[2] = result[2].fma(a2_sized, b_micropanel);
    result[6] = result[6].fma(a6_sized, b_micropanel);
    result[3] = result[3].fma(a3_sized, b_micropanel);
    result[7] = result[7].fma(a7_sized, b_micropanel);

    *result
}

unsafe fn kernel(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = F32x8::load(c_micropanel, mr);
    let mut c1 = F32x8::load(c_micropanel.add(m), mr);
    let mut c2 = F32x8::load(c_micropanel.add(2 * m), mr);
    let mut c3 = F32x8::load(c_micropanel.add(3 * m), mr);
    let mut c4 = F32x8::load(c_micropanel.add(4 * m), mr);
    let mut c5 = F32x8::load(c_micropanel.add(5 * m), mr);
    let mut c6 = F32x8::load(c_micropanel.add(6 * m), mr);
    let mut c7 = F32x8::load(c_micropanel.add(7 * m), mr);

    for k in 0..kc {
        let a_micropanel = F32x8::from(a_panel.data[k].as_slice());
        let b_micropanel = F32x8::from(b_panel.data[k].as_slice());

        // Duplicate 128-bit lanes to prepare for element broadcasting
        // permute2f128 with mask 0x00: [a0,a1,a2,a3, a0,a1,a2,a3]
        // permute2f128 with mask 0x11: [a4,a5,a6,a7, a4,a5,a6,a7]
        let a_lower_lane = a_micropanel.permute2f128::<0x00>();
        let a_upper_lane = a_micropanel.permute2f128::<0x11>();

        // First batch: broadcast elements 0 and 4, then 1 and 5
        // Interleave operations to maximize port utilization on modern CPUs
        let a0_broadcast = a_lower_lane.permute::<0x00>(); // [a0, a0, a0, a0, a0, a0, a0, a0]
        let a4_broadcast = a_upper_lane.permute::<0x00>(); // [a4, a4, a4, a4, a4, a4, a4, a4]
        let a1_broadcast = a_lower_lane.permute::<0x55>(); // [a1, a1, a1, a1, a1, a1, a1, a1]
        let a5_broadcast = a_upper_lane.permute::<0x55>(); // [a5, a5, a5, a5, a5, a5, a5, a5]

        // Create size-matched broadcast vectors once for all operations
        let a0_sized = a0_broadcast;
        let a4_sized = a4_broadcast;
        let a1_sized = a1_broadcast;
        let a5_sized = a5_broadcast;

        c0 = c0.fma(a0_sized, b_micropanel);
        c4 = c4.fma(a4_sized, b_micropanel);
        c1 = c1.fma(a1_sized, b_micropanel);
        c5 = c5.fma(a5_sized, b_micropanel);

        // Second batch: broadcast elements 2 and 6, then 3 and 7
        let a2_broadcast = a_lower_lane.permute::<0xAA>(); // [a2, a2, a2, a2, a2, a2, a2, a2]
        let a6_broadcast = a_upper_lane.permute::<0xAA>(); // [a6, a6, a6, a6, a6, a6, a6, a6]
        let a3_broadcast = a_lower_lane.permute::<0xFF>(); // [a3, a3, a3, a3, a3, a3, a3, a3]
        let a7_broadcast = a_upper_lane.permute::<0xFF>(); // [a7, a7, a7, a7, a7, a7, a7, a7]

        let a2_sized = a2_broadcast;
        let a6_sized = a6_broadcast;
        let a3_sized = a3_broadcast;
        let a7_sized = a7_broadcast;

        c2 = c2.fma(a2_sized, b_micropanel);
        c6 = c6.fma(a6_sized, b_micropanel);
        c3 = c3.fma(a3_sized, b_micropanel);
        c7 = c7.fma(a7_sized, b_micropanel);
    }

    c0.store_at(c_micropanel);
    c1.store_at(c_micropanel.add(m));
    c2.store_at(c_micropanel.add(2 * m));
    c3.store_at(c_micropanel.add(3 * m));
    c4.store_at(c_micropanel.add(4 * m));
    c5.store_at(c_micropanel.add(5 * m));
    c6.store_at(c_micropanel.add(6 * m));
    c7.store_at(c_micropanel.add(7 * m));
}

/// Create a test matrix in column-major format
fn create_test_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let idx = col * rows + row; // column-major indexing
                                        // matrix[idx] = rng.random_range(-1.0..1.0);
            matrix[idx] = (idx) as f32;
        }
    }
    matrix
}

fn main() {
    let (m, n, k) = (16, 16, 16);

    // let mc_values: Vec<usize> = (256..=512).step_by(128).collect(); // [64, 128, 192, ..., 1024]
    // let nc_values: Vec<usize> = (4096..=8192).step_by(1024).collect(); // [64, 128, 192, ..., 1024]

    let mc_values: Vec<usize> = vec![16]; // [64, 128, 192, ..., 1024]
    let nc_values: Vec<usize> = vec![16]; // [64, 128, 192, ..., 1024]

    // Create test matrices once
    let mut rng = StdRng::seed_from_u64(42);
    let a = create_test_matrix(m, k, &mut rng);
    let b = create_test_matrix(k, n, &mut rng);
    let mut c = create_test_matrix(m, n, &mut rng);
    // let mut c = vec![0.0; m * n];

    display_matrix_column_major(m, n, m, &c);

    for mc in mc_values.as_slice() {
        for nc in nc_values.as_slice() {
            // Enable performance profiling
            time_graph::enable_data_collection(true);

            println!("Size ={}x{}x{}, MC:{}, NC:{}, KC:{}", m, n, k, mc, nc, KC);
            matmul_with_params(&a, &b, &mut c, m, n, k, *mc, *nc);
            println!();

            // Get and print the performance profiling results
            let graph = time_graph::get_full_graph();
            // The following output formats are available but commented out:
            // println!("{}", graph.as_dot()); // DOT format for visualization
            // println!("{}", graph.as_json());     // JSON format
            println!("{}", graph.as_table()); // Full table
                                              // println!("{}", graph.as_short_table()); // Condensed table
        }
    }
}
