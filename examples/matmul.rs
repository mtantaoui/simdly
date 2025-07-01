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

pub fn display_matrix(m: usize, n: usize, ld: usize, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
    println!("---");
}

pub fn display_matrix_column_major(m: usize, n: usize, ld: usize, a: &[f32]) {
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

const A_PERMUTATION: i32 = 0b10_11_00_01;
const B_PERMUTATION: i32 = 0b01_00_11_10;

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

    let mut c0 = F32x8::load(c_micropanel, mr);
    let mut c1 = F32x8::load(c_micropanel.add(m), mr);
    let mut c2 = F32x8::load(c_micropanel.add(2 * m), mr);
    let mut c3 = F32x8::load(c_micropanel.add(3 * m), mr);
    let mut c4 = F32x8::load(c_micropanel.add(4 * m), mr);
    let mut c5 = F32x8::load(c_micropanel.add(5 * m), mr);
    let mut c6 = F32x8::load(c_micropanel.add(6 * m), mr);
    let mut c7 = F32x8::load(c_micropanel.add(7 * m), mr);

    for p in 0..kc {
        // load A micropanel
        let a = F32x8::load_aligned(a_panel.data[p].as_ptr());

        // load B micropanel
        let b = F32x8::load_aligned(b_panel.data[p].as_ptr());

        c0 = c0.fmadd(a, b);

        // let a_p = _mm256_permute_ps(a.elements, A_PERMUTATION);
        let a_p = a.permute::<A_PERMUTATION>();

        c1 = c1.fmadd(a_p, b);

        // let b_p0 = _mm256_permute_ps(b.elements, B_PERMUTATION);
        let b_p0 = b.permute::<B_PERMUTATION>();

        c2 = c2.fmadd(a, b_p0);
        c3 = c3.fmadd(a_p, b_p0);

        // let b_p1 = _mm256_permute2f128_ps(b.elements, b.elements, 0x01);
        let b_p1 = b.permute2f128::<0x01>(b);

        c4 = c4.fmadd(a, b_p1);
        c5 = c5.fmadd(a_p, b_p1);

        let b_p2 = b_p1.permute::<B_PERMUTATION>();

        c6 = c6.fmadd(a, b_p2);
        c7 = c7.fmadd(a_p, b_p2);
    }

    // Unshuffling and storing section
    // After computation, we have 8 accumulators that need to be reorganized into proper columns
    //
    // The computation creates these patterns:
    // c0: diagonal elements [a0*b0, a1*b1, a2*b2, a3*b3, a4*b4, a5*b5, a6*b6, a7*b7].
    // c1: A-permuted [a2*b0, a3*b1, a0*b2, a1*b3, a6*b4, a7*b5, a4*b6, a5*b7].
    // c2: B-permuted [a0*b1, a1*b0, a2*b3, a3*b2, a4*b5, a5*b4, a6*b7, a7*b6].
    // c3: both permuted [a2*b1, a3*b0, a0*b3, a1*b2, a6*b5, a7*b4, a4*b7, a5*b6].
    // c4: B 128-swapped [a0*b4, a1*b5, a2*b6, a3*b7, a4*b0, a5*b1, a6*b2, a7*b3].
    // c5: A-perm + B 128-swap [a2*b4, a3*b5, a0*b6, a1*b7, a6*b0, a7*b1, a4*b2, a5*b3].
    // c6: B-perm + B 128-swap [a0*b5, a1*b4, a2*b7, a3*b6, a4*b1, a5*b0, a6*b3, a7*b2]
    // c7: all perms [a2*b5, a3*b4, a0*b7, a1*b6, a6*b1, a7*b0, a4*b3, a5*b2]

    // Target: Column j should contain [a0*bj, a1*bj, a2*bj, a3*bj, a4*bj, a5*bj, a6*bj, a7*bj]

    // Strategy: Use 8x8 matrix transpose to reorganize the data efficiently
    // We can view c0-c7 as 8 rows that need to be transposed to get the 8 columns

    // // Step 1: Interleave adjacent pairs (32-bit granularity)
    // let r0 = c0.unpacklo(c1); // [c0[0],c1[0],c0[1],c1[1],c0[4],c1[4],c0[5],c1[5]]
    // let r1 = c0.unpackhi(c1); // [c0[2],c1[2],c0[3],c1[3],c0[6],c1[6],c0[7],c1[7]]
    // let r2 = c2.unpacklo(c3); // [c2[0],c3[0],c2[1],c3[1],c2[4],c3[4],c2[5],c3[5]]
    // let r3 = c2.unpackhi(c3); // [c2[2],c3[2],c2[3],c3[3],c2[6],c3[6],c2[7],c3[7]]
    // let r4 = c4.unpacklo(c5); // [c4[0],c5[0],c4[1],c5[1],c4[4],c5[4],c4[5],c5[5]]
    // let r5 = c4.unpackhi(c5); // [c4[2],c5[2],c4[3],c5[3],c4[6],c5[6],c4[7],c5[7]]
    // let r6 = c6.unpacklo(c7); // [c6[0],c7[0],c6[1],c7[1],c6[4],c7[4],c6[5],c7[5]]
    // let r7 = c6.unpackhi(c7); // [c6[2],c7[2],c6[3],c7[3],c6[6],c7[6],c6[7],c7[7]]

    // // Step 2: Interleave 64-bit pairs
    // let s0 = r0.unpacklo(r2); // [c0[0],c2[0],c1[0],c3[0],c0[4],c2[4],c1[4],c3[4]]
    // let s1 = r0.unpackhi(r2); // [c0[1],c2[1],c1[1],c3[1],c0[5],c2[5],c1[5],c3[5]]
    // let s2 = r1.unpacklo(r3); // [c0[2],c2[2],c1[2],c3[2],c0[6],c2[6],c1[6],c3[6]]
    // let s3 = r1.unpackhi(r3); // [c0[3],c2[3],c1[3],c3[3],c0[7],c2[7],c1[7],c3[7]]
    // let s4 = r4.unpacklo(r6); // [c4[0],c6[0],c5[0],c7[0],c4[4],c6[4],c5[4],c7[4]]
    // let s5 = r4.unpackhi(r6); // [c4[1],c6[1],c5[1],c7[1],c4[5],c6[5],c5[5],c7[5]]
    // let s6 = r5.unpacklo(r7); // [c4[2],c6[2],c5[2],c7[2],c4[6],c6[6],c5[6],c7[6]]
    // let s7 = r5.unpackhi(r7); // [c4[3],c6[3],c5[3],c7[3],c4[7],c6[7],c5[7],c7[7]]

    // // Step 3: Interleave 128-bit lanes to complete the transpose
    // let t0 = s0.permute2f128::<0x20>(s4); // [c0[0],c2[0],c1[0],c3[0],c4[0],c6[0],c5[0],c7[0]]
    // let t1 = s1.permute2f128::<0x20>(s5); // [c0[1],c2[1],c1[1],c3[1],c4[1],c6[1],c5[1],c7[1]]
    // let t2 = s2.permute2f128::<0x20>(s6); // [c0[2],c2[2],c1[2],c3[2],c4[2],c6[2],c5[2],c7[2]]
    // let t3 = s3.permute2f128::<0x20>(s7); // [c0[3],c2[3],c1[3],c3[3],c4[3],c6[3],c5[3],c7[3]]
    // let t4 = s0.permute2f128::<0x31>(s4); // [c0[4],c2[4],c1[4],c3[4],c4[4],c6[4],c5[4],c7[4]]
    // let t5 = s1.permute2f128::<0x31>(s5); // [c0[5],c2[5],c1[5],c3[5],c4[5],c6[5],c5[5],c7[5]]
    // let t6 = s2.permute2f128::<0x31>(s6); // [c0[6],c2[6],c1[6],c3[6],c4[6],c6[6],c5[6],c7[6]]
    // let t7 = s3.permute2f128::<0x31>(s7); // [c0[7],c2[7],c1[7],c3[7],c4[7],c6[7],c5[7],c7[7]]
}

pub fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Pre-condition checks for memory safety and correctness.
    // Panicking with a clear message is better than causing an out-of-bounds access.
    assert_eq!(a.len(), m * k, "Matrix A has incorrect dimensions");
    assert_eq!(b.len(), k * n, "Matrix B has incorrect dimensions");
    assert_eq!(c.len(), m * n, "Matrix C has incorrect dimensions");

    // The standard matrix multiplication algorithm.
    // Loop ordering is i, j, p which is straightforward but not the most cache-efficient.
    for i in 0..m {
        // Iterate over rows of C (and A)
        for j in 0..n {
            // Iterate over columns of C (and B)
            // Calculate the dot product of the i-th row of A and the j-th column of B.
            let mut dot_product = 0.0;
            for p in 0..k {
                // Iterate over the common dimension K
                // A[i, p] * B[p, j]
                // In row-major layout:
                // A[i, p] is at index i * (num_cols_A) + p = i * k + p
                // B[p, j] is at index p * (num_cols_B) + j = p * n + j
                dot_product += a[i * k + p] * b[p * n + j];
            }
            // Add the result to C: C[i, j] += dot_product
            // C[i, j] is at index i * (num_cols_C) + j = i * n + j
            c[i * n + j] += dot_product;
        }
    }
}

fn main() {
    let m = 8;
    let k = 8;
    let n = 8;

    let a = (0..m * k).map(|i| i as f32).collect::<Vec<f32>>();

    let b = (0..k * n).map(|i| i as f32).collect::<Vec<f32>>();

    let mut c_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();
    let mut c_naive_matmul = (0..m * n).map(|i| 0 as f32).collect::<Vec<f32>>();

    display_matrix(m, n, m, c_matmul.as_slice());

    matmul(&a, &b, &mut c_matmul, m, n, k);
    naive_matmul(&a, &b, &mut c_naive_matmul, m, n, k);

    display_matrix(m, n, m, c_matmul.as_slice());
    display_matrix(m, n, m, c_naive_matmul.as_slice());
}
