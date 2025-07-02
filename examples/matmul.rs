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

/// Packs a block from a column-major matrix `b` into a structured, aligned `BBlock` (KCxNC).
///
/// This is the highly efficient version that:
/// 1. Performs a single, aligned memory allocation for the entire block.
/// 2. Writes data directly into its final destination.
/// 3. Returns a structured `BBlock` that manages the memory's lifecycle.
///
/// # Generic Parameters
/// * `KC`: The packing dimension for rows (a compile-time constant).
/// * `NR`: The register-blocking parameter for columns (a compile-time constant).
///
/// # Arguments
/// * `b`: Slice representing the source column-major matrix `B`.
/// * `nc`: The number of columns from `b` to pack (runtime variable).
/// * `kc`: .
/// * `k`: The leading dimension (total number of rows) of the source matrix `b`.
pub fn pack_b<const KC: usize, const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,
) -> BBlock<KC, NR> {
    // 1. Allocate the BBlock. This gives us perfectly aligned, zeroed memory.
    let mut packed_block = BBlock::<KC, NR>::new(nc).expect("Memory allocation failed");

    // Iterate over the columns of B in panels of width NR.
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        // Get a mutable reference to the destination panel within the BBlock.
        let dest_panel = &mut packed_block[panel_idx];

        // The number of columns in this specific panel (can be < NR at the matrix edge).
        let nr_cols_in_panel = min(NR, nc - j_panel_start);

        // 2. Fill the panel. The data is packed into a row-major format.
        for p_row in 0..kc {
            for j_col_in_panel in 0..nr_cols_in_panel {
                // Calculate source index in column-major matrix B.
                let src_col = j_panel_start + j_col_in_panel;
                let src_idx = src_col * k + p_row;

                // Write directly into the final destination.
                dest_panel.data[p_row][j_col_in_panel] = b[src_idx];
            }
            // The remaining columns in `dest_panel.data[p_row]` (from nr_cols_in_panel..NR)
            // are already zero because we used `alloc_zeroed`, so padding is handled.
        }
    }

    packed_block
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

/// Packs a block from a column-major matrix `a` into an `ABlock` using `copy_from_slice`.
///
/// This version is highly efficient as it copies contiguous blocks of memory from the
/// source matrix directly into the contiguous columns of the destination panel.
pub fn pack_a<const MR: usize, const KC: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,
) -> ABlock<MR, KC> {
    let mut packed_block = ABlock::<MR, KC>::new(mc).expect("Memory allocation failed");

    // Iterate over A in row-panels of height MR.
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let mr_in_panel = min(MR, mc - i_panel_start);

        // Iterate through columns of the source block to fill the columns of the dest panel.
        for p_col in 0..kc {
            // 1. Define the source slice. This is a contiguous segment
            //    from a column of the original matrix `a`.
            let src_start = p_col * m + i_panel_start;
            let src_slice = &a[src_start..src_start + mr_in_panel];

            // 2. Define the destination slice. This is the start of the
            //    destination column in the packed panel.
            let dest_slice = &mut dest_panel.data[p_col][0..mr_in_panel];

            // 3. Perform the copy. This will be optimized to a single `memcpy`.
            dest_slice.copy_from_slice(src_slice);

            // Padding is already handled as the memory was zero-allocated.
            // The remaining elements in `dest_panel.data[p_col]` are untouched and remain zero.
        }
    }

    packed_block
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.chunks_mut(m * NC).enumerate().for_each(|(j, c_chunk)| {
        let jc = j * NC;
        let nc = min(NC, n - jc);

        a.chunks(m * KC).enumerate().for_each(|(p, a_chunk)| {
            let pc = p * KC;
            let kc = min(KC, k - pc);

            let b_block = pack_b::<KC, NR>(b, nc, kc, k);

            for ic in (0..m).step_by(MC) {
                let mc = min(MC, m - ic);

                let a_block = pack_a::<MR, KC>(&a_chunk[ic..], mc, kc, m);

                b_block
                    .as_panels()
                    .iter()
                    .enumerate()
                    .for_each(|(jr, b_panel)| {
                        a_block
                            .as_panels()
                            .iter()
                            .enumerate()
                            .for_each(|(ir, a_panel)| {
                                let nr = min(NR, nc - jr * NR);
                                let mr = min(MR, mc - ir * MR);

                                println!("C micro panel, Size: {}x{}", mr, nr);

                                let c_micropanel =
                                    c_chunk[(jr * NR * m + (ic + ir * MR))..].as_mut_ptr();

                                unsafe {
                                    kernel_8x8(a_panel, b_panel, c_micropanel, mr, nr, kc, m)
                                };
                                // display_matrix(mr, nr, m, c_micropanel);
                            })
                    });
            }
        });
    });
}

const A_PERMUTATION: i32 = 0b01_00_11_10;
const B_PERMUTATION: i32 = 0x01;

const SHUFFLE_MASK: i32 = 0b11100100;

const FINAL_PERMUTATION_1: i32 = 0x20;
const FINAL_PERMUTATION_2: i32 = 0x31;

unsafe fn kernel_8x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    println!("C micro panel, Size: {}x{}", mr, nr);

    // Only when Using column major storage
    let (a_panel, b_panel) = (b_panel, a_panel);

    let mut ab_striped0: F32x8 = F32x8::zero();
    let mut ab_striped1: F32x8 = F32x8::zero();
    let mut ab_striped2: F32x8 = F32x8::zero();
    let mut ab_striped3: F32x8 = F32x8::zero();
    let mut ab_striped4: F32x8 = F32x8::zero();
    let mut ab_striped5: F32x8 = F32x8::zero();
    let mut ab_striped6: F32x8 = F32x8::zero();
    let mut ab_striped7: F32x8 = F32x8::zero();

    for p in 0..kc {
        let a_col = F32x8::load_aligned(a_panel.data[p].as_ptr());

        let a_even = a_col.moveldup();

        let a_odd = a_col.movehdup();

        let a_even_permuted = a_even.permute::<A_PERMUTATION>();

        let a_odd_permuted = a_odd.permute::<A_PERMUTATION>();

        let b_row = F32x8::load_aligned(b_panel.data[p].as_ptr());

        let b_row_permuted = b_row.permute2f128::<B_PERMUTATION>(b_row);

        ab_striped0.fmadd(a_even, b_row);
        ab_striped1.fmadd(a_even_permuted, b_row);
        ab_striped2.fmadd(a_even, b_row_permuted);
        ab_striped3.fmadd(a_even_permuted, b_row_permuted);

        ab_striped4.fmadd(a_odd, b_row);
        ab_striped5.fmadd(a_odd_permuted, b_row);
        ab_striped6.fmadd(a_odd, b_row_permuted);
        ab_striped7.fmadd(a_odd_permuted, b_row_permuted);
    }

    // Shuffle within vectors
    let pair_0_1 = ab_striped0.shuffle::<SHUFFLE_MASK>(ab_striped1);
    let pair_1_0 = ab_striped1.shuffle::<SHUFFLE_MASK>(ab_striped0);

    let pair_2_3 = ab_striped2.shuffle::<SHUFFLE_MASK>(ab_striped3);
    let pair_3_2 = ab_striped3.shuffle::<SHUFFLE_MASK>(ab_striped2);

    let pair_4_5 = ab_striped4.shuffle::<SHUFFLE_MASK>(ab_striped5);
    let pair_5_4 = ab_striped5.shuffle::<SHUFFLE_MASK>(ab_striped4);

    let pair_6_7 = ab_striped6.shuffle::<SHUFFLE_MASK>(ab_striped7);
    let pair_7_6 = ab_striped7.shuffle::<SHUFFLE_MASK>(ab_striped6);

    let c_col_0 = pair_0_1.permute2f128::<FINAL_PERMUTATION_1>(pair_2_3);
    let c_col_1 = pair_4_5.permute2f128::<FINAL_PERMUTATION_1>(pair_6_7);
    let c_col_2 = pair_1_0.permute2f128::<FINAL_PERMUTATION_1>(pair_3_2);
    let c_col_3 = pair_5_4.permute2f128::<FINAL_PERMUTATION_1>(pair_7_6);
    let c_col_4 = pair_0_1.permute2f128::<FINAL_PERMUTATION_2>(pair_2_3);
    let c_col_5 = pair_4_5.permute2f128::<FINAL_PERMUTATION_2>(pair_6_7);
    let c_col_6 = pair_1_0.permute2f128::<FINAL_PERMUTATION_2>(pair_3_2);
    let c_col_7 = pair_5_4.permute2f128::<FINAL_PERMUTATION_2>(pair_7_6);

    println!("C0: {:?}", c_col_0);
    println!("C1: {:?}", c_col_1);
    println!("C2: {:?}", c_col_2);
    println!("C3: {:?}", c_col_3);
    println!("C4: {:?}", c_col_4);
    println!("C5: {:?}", c_col_5);
    println!("C6: {:?}", c_col_6);
    println!("C7: {:?}", c_col_7);

    // Store the results back to the micro panel.

    match mr {
        MR => {
            // // For 8 rows, we can store directly.
            // c_col_0.store_at(c_micropanel.add(0));
            // c_col_1.store_at(c_micropanel.add(m));
            // c_col_2.store_at(c_micropanel.add(m * 2));
            // c_col_3.store_at(c_micropanel.add(m * 3));
            // c_col_4.store_at(c_micropanel.add(m * 4));
            // c_col_5.store_at(c_micropanel.add(m * 5));
            // c_col_6.store_at(c_micropanel.add(m * 6));
            // c_col_7.store_at(c_micropanel.add(m * 7));
        }
        _ => {
            // // For 8 rows, we can store directly.
            // c_col_0.store_at_partial(c_micropanel.add(0));
            // c_col_1.store_at_partial(c_micropanel.add(m));
            // c_col_2.store_at_partial(c_micropanel.add(m * 2));
            // c_col_3.store_at_partial(c_micropanel.add(m * 3));
            // c_col_4.store_at_partial(c_micropanel.add(m * 4));
            // c_col_5.store_at_partial(c_micropanel.add(m * 5));
            // c_col_6.store_at_partial(c_micropanel.add(m * 6));
            // c_col_7.store_at_partial(c_micropanel.add(m * 7));
        }
    }
}

fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    println!("A: {:?}", a);
    println!("B: {:?}", b);
    println!("C: {:?}", c);

    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                // Column-major index: i + j * rows
                let a_idx = row + inner * m;
                let b_idx = inner + col * k;
                sum += a[a_idx] * b[b_idx];
            }
            let c_idx = row + col * m;
            c[c_idx] = sum;
        }
    }
}

fn main() {
    let m = 2;
    let k = 2;
    let n = 2;

    let a = (0..m * k).map(|i| i as f32).collect::<Vec<f32>>();
    let b = (0..k * n).map(|i| i as f32).collect::<Vec<f32>>();

    let mut c_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();
    let mut c_naive_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();

    // display_matrix_column_major(m, n, m, c_naive_matmul.as_slice());
    // display_matrix_row_major(m, n, m, c_naive_matmul.as_slice());

    matmul(&a, &b, &mut c_matmul, m, n, k);
    naive_matmul(&a, &b, &mut c_naive_matmul, m, n, k);

    // display_matrix_column_major(m, n, m, c_naive_matmul.as_slice());
    display_matrix_column_major(m, n, m, c_matmul.as_slice());
    display_matrix_column_major(m, n, m, c_naive_matmul.as_slice());
}
