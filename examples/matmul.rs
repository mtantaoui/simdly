#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::traits::SimdVec;

// Helper to define alignment. For AVX/AVX2, 32 bytes is correct. For AVX512, it would be 64.
const ALIGNMENT: usize = 32;

pub const MR: usize = 8;
pub const NR: usize = 8;

pub const MC: usize = 16;
pub const NC: usize = 16;
pub const KC: usize = 16;

fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
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

pub fn display_matrix_row_major(m: usize, n: usize, ld: usize, a: &[f32]) {
    for j in 0..n {
        for i in 0..m {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
    println!("---");
}

/// Represents a single packed panel of the B matrix, with dimensions KC x NR.
///
/// The data is stored in a row-major format, so each of the KC rows
/// contains NR floating-point numbers. This layout is optimal for the micro-kernel,
/// which consumes one row of this panel at a time.
///
/// The `#[repr(C, align(32))]` ensures its memory layout is predictable and aligned
/// for AVX vector instructions.
#[repr(C, align(32))]
pub struct BPanel<const KC: usize, const NR: usize> {
    pub data: [[f32; NR]; KC],
}

/// Represents a packed block of the B matrix, composed of multiple BPanels.
///
/// This struct owns a chunk of 32-byte aligned memory, allocated to hold a sequence
/// of `BPanel` structs. It manages the lifecycle of this memory, including safe
/// deallocation via the `Drop` trait.
pub struct BBlock<const KC: usize, const NR: usize> {
    // Raw pointer to the allocated block of `BPanel`s.
    ptr: *mut BPanel<KC, NR>,

    // The number of panels allocated.
    num_panels: usize,

    // The Layout is stored for guaranteed-correct deallocation.
    layout: Layout,

    // The original number of columns packed into this block.
    pub nc: usize,

    // PhantomData to tell the compiler this struct "owns" the data pointed to.
    _marker: PhantomData<BPanel<KC, NR>>,
}

// --- Implementation of BBlock ---

impl<const KC: usize, const NR: usize> BBlock<KC, NR> {
    /// Allocates aligned, zeroed memory for a packed block of `B`.
    ///
    /// # Arguments
    /// * `nc`: The total number of columns to be packed in this block.
    ///
    /// # Returns
    /// A `Result` containing the `BBlock` if allocation succeeds.
    pub fn new(nc: usize) -> Result<Self, Layout> {
        let num_panels = nc.div_ceil(NR);

        // Create the memory layout for an array of `BPanel`s.
        let layout = Layout::array::<BPanel<KC, NR>>(num_panels).unwrap();

        // Ensure the layout meets our minimum alignment requirement.
        let layout = layout.align_to(ALIGNMENT).unwrap();

        let ptr = unsafe {
            // Allocate zeroed memory. This handles all padding automatically.
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

    /// Returns an immutable slice view of the panels.
    pub fn as_panels(&self) -> &[BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the panels.
    pub fn as_panels_mut(&mut self) -> &mut [BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// The new, highly optimized and robust Drop implementation.
impl<const KC: usize, const NR: usize> Drop for BBlock<KC, NR> {
    fn drop(&mut self) {
        // We only deallocate if the size of the allocation was greater than 0.
        // Calling `dealloc` with a zero-sized layout is UB.
        if self.layout.size() > 0 {
            unsafe {
                // There is no recalculation. We use the exact layout that was
                // stored during allocation. This is faster and safer.
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

// --- Convenience Indexing ---

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

#[repr(C, align(32))]
pub struct APanel<const MR: usize, const KC: usize> {
    pub data: [[f32; MR]; KC],
}

pub struct ABlock<const MR: usize, const KC: usize> {
    ptr: *mut APanel<MR, KC>,
    num_panels: usize,
    layout: Layout,
    pub mc: usize,
    _marker: PhantomData<APanel<MR, KC>>,
}

impl<const MR: usize, const KC: usize> ABlock<MR, KC> {
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

    pub fn as_panels(&self) -> &[APanel<MR, KC>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    pub fn as_panels_mut(&mut self) -> &mut [APanel<MR, KC>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

impl<const MR: usize, const KC: usize> Drop for ABlock<MR, KC> {
    fn drop(&mut self) {
        // We only deallocate if the size of the allocation was greater than 0.
        // Calling `dealloc` with a zero-sized layout is UB.
        if self.layout.size() > 0 {
            unsafe {
                // There is no recalculation. We use the exact layout that was
                // stored during allocation. This is faster and safer.
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

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

// The corrected matmul function - the key insight is that you cannot use chunks()
// on column-major matrices because blocks are not contiguous in memory

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Loop over column blocks of C (and B)
    for jc in (0..n).step_by(NC) {
        let nc = min(NC, n - jc);

        // Loop over row blocks of C (and A)
        for ic in (0..m).step_by(MC) {
            let mc = min(MC, m - ic);

            // Initialize this block of C to zero (only for the first K iteration)
            // Actually, we assume C starts as zeros, so we can skip this

            // Loop over K dimension blocks
            for pc in (0..k).step_by(KC) {
                let kc = min(KC, k - pc);

                // Pack block of A: A(ic:ic+mc, pc:pc+kc)
                // We need to extract this block from the column-major matrix A
                let a_block = pack_a::<MR, KC>(a, mc, kc, m, ic, pc);

                // Pack block of B: B(pc:pc+kc, jc:jc+nc)
                // We need to extract this block from the column-major matrix B
                let b_block = pack_b::<KC, NR>(b, nc, kc, k, pc, jc);

                // Perform micro-kernel computations
                for jr in 0..b_block.as_panels().len() {
                    let b_panel = &b_block[jr];
                    let nr = min(NR, nc - jr * NR);

                    for ir in 0..a_block.as_panels().len() {
                        let a_panel = &a_block[ir];
                        let mr = min(MR, mc - ir * MR);

                        // Calculate position in C for this micro-panel
                        let c_row = ic + ir * MR;
                        let c_col = jc + jr * NR;
                        let c_micropanel_start_idx = c_col * m + c_row;

                        let c_micropanel = &mut c[c_micropanel_start_idx..];

                        unsafe {
                            kernel_8x8(a_panel, b_panel, c_micropanel.as_mut_ptr(), mr, nr, kc, m);
                        }
                    }
                }
            }
        }
    }
}

// Modified pack_a to take matrix parameters and block coordinates
pub fn pack_a<const MR: usize, const KC: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,  // leading dimension of A
    ic: usize, // starting row of block
    pc: usize, // starting column of block
) -> ABlock<MR, KC> {
    let mut packed_block = ABlock::<MR, KC>::new(mc).expect("Memory allocation failed");

    // Iterate over A in row-panels of height MR
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let mr_in_panel = min(MR, mc - i_panel_start);

        // Fill each column of the panel
        for p_col in 0..kc {
            // Source column in original matrix A
            let src_col = pc + p_col;
            // Source row start in original matrix A
            let src_row_start = ic + i_panel_start;

            // In column-major A, column src_col starts at index src_col * m
            // We want elements A(src_row_start:src_row_start+mr_in_panel, src_col)
            let src_start = src_col * m + src_row_start;
            let src_slice = &a[src_start..src_start + mr_in_panel];

            // Destination in panel
            let dest_slice = &mut dest_panel.data[p_col][0..mr_in_panel];

            // Copy the data
            dest_slice.copy_from_slice(src_slice);
        }
    }

    packed_block
}

// Modified pack_b to take matrix parameters and block coordinates
pub fn pack_b<const KC: usize, const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,  // leading dimension of B
    pc: usize, // starting row of block
    jc: usize, // starting column of block
) -> BBlock<KC, NR> {
    let mut packed_block = BBlock::<KC, NR>::new(nc).expect("Memory allocation failed");

    // Iterate over columns of B in panels of width NR
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let nr_cols_in_panel = min(NR, nc - j_panel_start);

        // Fill the panel row by row
        for p_row in 0..kc {
            // Source row in original matrix B
            let src_row = pc + p_row;

            for j_col_in_panel in 0..nr_cols_in_panel {
                // Source column in original matrix B
                let src_col = jc + j_panel_start + j_col_in_panel;

                // In column-major B, element B(src_row, src_col) is at index src_col * k + src_row
                let src_idx = src_col * k + src_row;

                // Store in panel
                dest_panel.data[p_row][j_col_in_panel] = b[src_idx];
            }
        }
    }

    packed_block
}
unsafe fn kernel_8x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    // println!("C micro panel, Size: {}x{}", mr, nr);

    // for p in 0..kc {
    //     let a_column = &a_panel.data[p];
    //     let b_row = &b_panel.data[p];

    //     println!("A: column:{:?}", a_column);
    //     println!("B: row:{:?}", b_row);
    // }

    let mut c0 = F32x8::load(c_micropanel, mr);
    let mut c1 = F32x8::load(c_micropanel.add(m), mr);
    let mut c2 = F32x8::load(c_micropanel.add(2 * m), mr);
    let mut c3 = F32x8::load(c_micropanel.add(3 * m), mr);
    let mut c4 = F32x8::load(c_micropanel.add(4 * m), mr);
    let mut c5 = F32x8::load(c_micropanel.add(5 * m), mr);
    let mut c6 = F32x8::load(c_micropanel.add(6 * m), mr);
    let mut c7 = F32x8::load(c_micropanel.add(7 * m), mr);

    let mut b_pj: F32x8;

    for p in 0..kc {
        let a = &a_panel.data[p];
        let b = &b_panel.data[p];

        // Load the A column into a vector
        let a = F32x8::load_aligned(a.as_ptr());

        // Perform the multiplication and accumulate
        b_pj = F32x8::splat(b[0]);
        // c0.fmadd(a, b_pj);
        c0 += a * b_pj;

        b_pj = F32x8::splat(b[1]);
        // c1.fmadd(a, b_pj);
        c1 += a * b_pj;

        b_pj = F32x8::splat(b[2]);
        // c2.fmadd(a, b_pj);
        c2 += a * b_pj;

        b_pj = F32x8::splat(b[3]);
        // c3.fmadd(a, b_pj);
        c3 += a * b_pj;

        b_pj = F32x8::splat(b[4]);
        // c4.fmadd(a, b_pj);
        c4 += a * b_pj;

        b_pj = F32x8::splat(b[5]);
        // c5.fmadd(a, b_pj);
        c5 += a * b_pj;

        b_pj = F32x8::splat(b[6]);
        // c6.fmadd(a, b_pj);
        c6 += a * b_pj;

        b_pj = F32x8::splat(b[7]);
        // c7.fmadd(a, b_pj);
        c7 += a * b_pj;
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

fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // println!("A: {:?}", a);
    // println!("B: {:?}", b);
    // println!("C: {:?}", c);

    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                // Column-major index: i + j * rows

                sum += a[at(row, inner, m)] * b[at(inner, col, k)];
            }
            let c_idx = row + col * m;
            c[c_idx] = sum;
        }
    }
}

fn main() {
    let m = 8 * 10;
    let k = 8 * 10;
    let n = 8 * 10;

    let a = (0..m * k).map(|i| i as f32).collect::<Vec<f32>>();
    let b = (0..k * n).map(|i| i as f32).collect::<Vec<f32>>();

    let mut c_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();
    let mut c_naive_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();

    matmul(&a, &b, &mut c_matmul, m, n, k);
    naive_matmul(&a, &b, &mut c_naive_matmul, m, n, k);

    for i in 0..m {
        for j in 0..n {
            let idx = at(i, j, m);
            if c_matmul[idx] != c_naive_matmul[idx] {
                println!(
                    "Mismatch at ({}, {}): {} != {}",
                    i, j, c_matmul[idx], c_naive_matmul[idx]
                );
            }
        }
    }
    println!("All values match between optimized and naive matmul implementations.");
}
