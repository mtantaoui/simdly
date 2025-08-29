//! Optimized matrix multiplication microkernels using AVX2 SIMD instructions.
//!
//! This module contains high-performance microkernels that form the computational core
//! of the BLIS-style matrix multiplication algorithm. These kernels operate on small
//! matrix blocks (typically 8×8 or 16×8) and are optimized for AVX2's 256-bit vectors.

use std::cmp::min;

use crate::simd::{
    avx2::{
        f32x8::{F32x8, LANE_COUNT},
        panels::{APanel, BPanel, KC, MR, NR},
    },
    SimdLoad, SimdMath, SimdShuffle, SimdStore,
};

/// Helper function to load C matrix elements, choosing between full and partial load.
///
/// Automatically selects between aligned/unaligned full loads and masked partial loads
/// based on the number of valid elements to prevent buffer overruns.
#[inline(always)]
pub(crate) unsafe fn load_c(ptr: *const f32, size: usize) -> F32x8 {
    let size = min(size, LANE_COUNT);
    if size == LANE_COUNT {
        F32x8::load(ptr, size)
    } else {
        F32x8::load_partial(ptr, size)
    }
}

/// AVX2 microkernel for 8×8 matrix multiplication using FMA operations.
///
/// This is the core computational kernel that multiplies an 8×KC block of matrix A
/// with a KC×8 block of matrix B, accumulating the result into an 8×8 block of matrix C.
/// Uses register tiling with 8 AVX2 registers to maximize arithmetic intensity.
///
/// # Arguments
/// * `a_panel` - Packed A matrix panel (8×KC elements)
/// * `b_panel` - Packed B matrix panel (KC×8 elements) 
/// * `c_micropanel` - Output C matrix block (column-major, 8×8 elements)
/// * `mr` - Actual number of rows in A block (≤ 8)
/// * `nr` - Actual number of columns in B block (≤ 8)
/// * `kc` - Inner dimension (number of terms in dot products)
/// * `m` - Leading dimension of matrix C (column stride)
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 8×8 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
pub(crate) unsafe fn kernel_8x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = load_c(c_micropanel, mr);
    let mut c1 = load_c(c_micropanel.add(m), mr);
    let mut c2 = load_c(c_micropanel.add(2 * m), mr);
    let mut c3 = load_c(c_micropanel.add(3 * m), mr);
    let mut c4 = load_c(c_micropanel.add(4 * m), mr);
    let mut c5 = load_c(c_micropanel.add(5 * m), mr);
    let mut c6 = load_c(c_micropanel.add(6 * m), mr);
    let mut c7 = load_c(c_micropanel.add(7 * m), mr);

    for k in 0..kc {
        let a_micropanel = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Use the proven B-broadcasting pattern with better instruction interleaving
        let b_lower_lane = b_micropanel.permute2f128::<0x00>();
        let b_upper_lane = b_micropanel.permute2f128::<0x11>();

        // First batch: broadcast elements 0 and 4, then 1 and 5
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        c0 = c0.fma(a_micropanel, b0_broadcast);
        c4 = c4.fma(a_micropanel, b4_broadcast);
        c1 = c1.fma(a_micropanel, b1_broadcast);
        c5 = c5.fma(a_micropanel, b5_broadcast);

        let b2_broadcast = b_lower_lane.permute::<0xAA>();
        let b6_broadcast = b_upper_lane.permute::<0xAA>();
        let b3_broadcast = b_lower_lane.permute::<0xFF>();
        let b7_broadcast = b_upper_lane.permute::<0xFF>();

        c2 = c2.fma(a_micropanel, b2_broadcast);
        c6 = c6.fma(a_micropanel, b6_broadcast);
        c3 = c3.fma(a_micropanel, b3_broadcast);
        c7 = c7.fma(a_micropanel, b7_broadcast);
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

/// AVX2 microkernel for 16×8 matrix multiplication (experimental).
///
/// Extended version of the 8×8 kernel that processes 16 rows of matrix A simultaneously.
/// This kernel uses twice as many registers (16 AVX2 registers) to potentially improve
/// performance by increasing register utilization and reducing memory traffic.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (16×KC elements, stored as two 8×KC blocks)
/// * `b_panel` - Packed B matrix panel (KC×8 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 16×8 elements)
/// * `mr` - Actual number of rows in A block (≤ 16)
/// * `nr` - Actual number of columns in B block (≤ 8) 
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Performance Notes
/// This kernel may not always outperform the 8×8 version due to register pressure
/// and increased complexity. Benchmark both versions for your specific workload.
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 16×8 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
pub(crate) unsafe fn kernel_16x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0_hi = load_c(c_micropanel, mr);
    let mut c0_lo = load_c(c_micropanel.add(LANE_COUNT), mr);

    let mut c1_hi = load_c(c_micropanel.add(m), mr);
    let mut c1_lo = load_c(c_micropanel.add(m + LANE_COUNT), mr);

    let mut c2_hi = load_c(c_micropanel.add(2 * m), mr);
    let mut c2_lo = load_c(c_micropanel.add(2 * m + LANE_COUNT), mr);

    let mut c3_hi = load_c(c_micropanel.add(3 * m), mr);
    let mut c3_lo = load_c(c_micropanel.add(3 * m + LANE_COUNT), mr);

    let mut c4_hi = load_c(c_micropanel.add(4 * m), mr);
    let mut c4_lo = load_c(c_micropanel.add(4 * m + LANE_COUNT), mr);

    let mut c5_hi = load_c(c_micropanel.add(5 * m), mr);
    let mut c5_lo = load_c(c_micropanel.add(5 * m + LANE_COUNT), mr);

    let mut c6_hi = load_c(c_micropanel.add(6 * m), mr);
    let mut c6_lo = load_c(c_micropanel.add(6 * m + LANE_COUNT), mr);

    let mut c7_hi = load_c(c_micropanel.add(7 * m), mr);
    let mut c7_lo = load_c(c_micropanel.add(7 * m + LANE_COUNT), mr);

    for k in 0..kc {
        // high

        let a_micropanel_hi = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Use the same interleaved broadcast pattern as simd_outer_product
        let b_lower_lane = b_micropanel.permute2f128::<0x00>();
        let b_upper_lane = b_micropanel.permute2f128::<0x11>();

        // First batch: broadcast elements 0 and 4, then 1 and 5
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        c0_hi = c0_hi.fma(a_micropanel_hi, b0_broadcast);
        c4_hi = c4_hi.fma(a_micropanel_hi, b4_broadcast);
        c1_hi = c1_hi.fma(a_micropanel_hi, b1_broadcast);
        c5_hi = c5_hi.fma(a_micropanel_hi, b5_broadcast);

        let b2_broadcast = b_lower_lane.permute::<0xAA>();
        let b6_broadcast = b_upper_lane.permute::<0xAA>();
        let b3_broadcast = b_lower_lane.permute::<0xFF>();
        let b7_broadcast = b_upper_lane.permute::<0xFF>();

        c2_hi = c2_hi.fma(a_micropanel_hi, b2_broadcast);
        c6_hi = c6_hi.fma(a_micropanel_hi, b6_broadcast);
        c3_hi = c3_hi.fma(a_micropanel_hi, b3_broadcast);
        c7_hi = c7_hi.fma(a_micropanel_hi, b7_broadcast);

        // low

        let a_micropanel_lo = F32x8::load_aligned(a_panel.data[k].as_ptr().add(LANE_COUNT));

        c0_lo = c0_lo.fma(a_micropanel_lo, b0_broadcast);
        c4_lo = c4_lo.fma(a_micropanel_lo, b4_broadcast);
        c1_lo = c1_lo.fma(a_micropanel_lo, b1_broadcast);
        c5_lo = c5_lo.fma(a_micropanel_lo, b5_broadcast);

        c2_lo = c2_lo.fma(a_micropanel_lo, b2_broadcast);
        c6_lo = c6_lo.fma(a_micropanel_lo, b6_broadcast);
        c3_lo = c3_lo.fma(a_micropanel_lo, b3_broadcast);
        c7_lo = c7_lo.fma(a_micropanel_lo, b7_broadcast);
    }

    c0_hi.store_at(c_micropanel);
    c0_lo.store_at(c_micropanel.add(LANE_COUNT));

    c1_hi.store_at(c_micropanel.add(m));
    c1_lo.store_at(c_micropanel.add(m + LANE_COUNT));

    c2_hi.store_at(c_micropanel.add(2 * m));
    c2_lo.store_at(c_micropanel.add(2 * m + LANE_COUNT));

    c3_hi.store_at(c_micropanel.add(3 * m));
    c3_lo.store_at(c_micropanel.add(3 * m + LANE_COUNT));

    c4_hi.store_at(c_micropanel.add(4 * m));
    c4_lo.store_at(c_micropanel.add(4 * m + LANE_COUNT));

    c5_hi.store_at(c_micropanel.add(5 * m));
    c5_lo.store_at(c_micropanel.add(5 * m + LANE_COUNT));

    c6_hi.store_at(c_micropanel.add(6 * m));
    c6_lo.store_at(c_micropanel.add(6 * m + LANE_COUNT));

    c7_hi.store_at(c_micropanel.add(7 * m));
    c7_lo.store_at(c_micropanel.add(7 * m + LANE_COUNT));
}

/// AVX2 microkernel for 24×8 matrix multiplication (experimental).
///
/// Extended microkernel processing 24 rows of matrix A (3×8 AVX2 vectors) with 8 columns
/// of matrix B. Uses 24 AVX2 registers for C accumulation, requiring careful register
/// management to avoid spilling.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (24×KC elements, stored as three 8×KC blocks)
/// * `b_panel` - Packed B matrix panel (KC×8 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 24×8 elements)
/// * `mr` - Actual number of rows in A block (≤ 24)
/// * `nr` - Actual number of columns in B block (≤ 8) 
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Warning
/// **POTENTIAL BUG**: This kernel accesses `a_panel.data[k][LANE_COUNT..]` but APanel.data[k]
/// is only [f32; MR=8], not [f32; 24]. This may cause buffer overruns or incorrect results.
/// The APanel layout may need to be redesigned for kernels larger than 8×8.
///
/// # Performance Notes
/// May experience register pressure. Profile against smaller kernels for your workload.
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 24×8 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
/// - **WARNING**: Current implementation may have memory access issues
pub(crate) unsafe fn kernel_24x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0_0 = load_c(c_micropanel, mr);
    let mut c1_0 = load_c(c_micropanel.add(m), mr);
    let mut c2_0 = load_c(c_micropanel.add(2 * m), mr);
    let mut c3_0 = load_c(c_micropanel.add(3 * m), mr);
    let mut c4_0 = load_c(c_micropanel.add(4 * m), mr);
    let mut c5_0 = load_c(c_micropanel.add(5 * m), mr);
    let mut c6_0 = load_c(c_micropanel.add(6 * m), mr);
    let mut c7_0 = load_c(c_micropanel.add(7 * m), mr);

    let mut c0_1 = load_c(c_micropanel.add(LANE_COUNT), mr);
    let mut c1_1 = load_c(c_micropanel.add(m + LANE_COUNT), mr);
    let mut c2_1 = load_c(c_micropanel.add(2 * m + LANE_COUNT), mr);
    let mut c3_1 = load_c(c_micropanel.add(3 * m + LANE_COUNT), mr);
    let mut c4_1 = load_c(c_micropanel.add(4 * m + LANE_COUNT), mr);
    let mut c5_1 = load_c(c_micropanel.add(5 * m + LANE_COUNT), mr);
    let mut c6_1 = load_c(c_micropanel.add(6 * m + LANE_COUNT), mr);
    let mut c7_1 = load_c(c_micropanel.add(7 * m + LANE_COUNT), mr);

    let mut c0_2 = load_c(c_micropanel.add(LANE_COUNT * 2), mr);
    let mut c1_2 = load_c(c_micropanel.add(m + LANE_COUNT * 2), mr);
    let mut c2_2 = load_c(c_micropanel.add(2 * m + LANE_COUNT * 2), mr);
    let mut c3_2 = load_c(c_micropanel.add(3 * m + LANE_COUNT * 2), mr);
    let mut c4_2 = load_c(c_micropanel.add(4 * m + LANE_COUNT * 2), mr);
    let mut c5_2 = load_c(c_micropanel.add(5 * m + LANE_COUNT * 2), mr);
    let mut c6_2 = load_c(c_micropanel.add(6 * m + LANE_COUNT * 2), mr);
    let mut c7_2 = load_c(c_micropanel.add(7 * m + LANE_COUNT * 2), mr);

    for k in 0..kc {
        // 0

        let a_micropanel_0 = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Use the same interleaved broadcast pattern as simd_outer_product
        let b_lower_lane = b_micropanel.permute2f128::<0x00>();
        let b_upper_lane = b_micropanel.permute2f128::<0x11>();

        // First batch: broadcast elements 0 and 4, then 1 and 5
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        c0_0 = c0_0.fma(a_micropanel_0, b0_broadcast);
        c4_0 = c4_0.fma(a_micropanel_0, b4_broadcast);
        c1_0 = c1_0.fma(a_micropanel_0, b1_broadcast);
        c5_0 = c5_0.fma(a_micropanel_0, b5_broadcast);

        let b2_broadcast = b_lower_lane.permute::<0xAA>();
        let b6_broadcast = b_upper_lane.permute::<0xAA>();
        let b3_broadcast = b_lower_lane.permute::<0xFF>();
        let b7_broadcast = b_upper_lane.permute::<0xFF>();

        c2_0 = c2_0.fma(a_micropanel_0, b2_broadcast);
        c6_0 = c6_0.fma(a_micropanel_0, b6_broadcast);
        c3_0 = c3_0.fma(a_micropanel_0, b3_broadcast);
        c7_0 = c7_0.fma(a_micropanel_0, b7_broadcast);

        // 1

        let a_micropanel_1 = F32x8::load_aligned(a_panel.data[k][LANE_COUNT..].as_ptr());

        c0_1 = c0_1.fma(a_micropanel_1, b0_broadcast);
        c4_1 = c4_1.fma(a_micropanel_1, b4_broadcast);
        c1_1 = c1_1.fma(a_micropanel_1, b1_broadcast);
        c5_1 = c5_1.fma(a_micropanel_1, b5_broadcast);

        c2_1 = c2_1.fma(a_micropanel_1, b2_broadcast);
        c6_1 = c6_1.fma(a_micropanel_1, b6_broadcast);
        c3_1 = c3_1.fma(a_micropanel_1, b3_broadcast);
        c7_1 = c7_1.fma(a_micropanel_1, b7_broadcast);

        // 2

        let a_micropanel_2 = F32x8::load_aligned(a_panel.data[k][2 * LANE_COUNT..].as_ptr());

        c0_2 = c0_2.fma(a_micropanel_2, b0_broadcast);
        c4_2 = c4_2.fma(a_micropanel_2, b4_broadcast);
        c1_2 = c1_2.fma(a_micropanel_2, b1_broadcast);
        c5_2 = c5_2.fma(a_micropanel_2, b5_broadcast);

        c2_2 = c2_2.fma(a_micropanel_2, b2_broadcast);
        c6_2 = c6_2.fma(a_micropanel_2, b6_broadcast);
        c3_2 = c3_2.fma(a_micropanel_2, b3_broadcast);
        c7_2 = c7_2.fma(a_micropanel_2, b7_broadcast);
    }

    c0_0.store_at(c_micropanel);
    c1_0.store_at(c_micropanel.add(m));
    c2_0.store_at(c_micropanel.add(2 * m));
    c3_0.store_at(c_micropanel.add(3 * m));
    c4_0.store_at(c_micropanel.add(4 * m));
    c5_0.store_at(c_micropanel.add(5 * m));
    c6_0.store_at(c_micropanel.add(6 * m));
    c7_0.store_at(c_micropanel.add(7 * m));

    c0_1.store_at(c_micropanel.add(LANE_COUNT));
    c1_1.store_at(c_micropanel.add(m + LANE_COUNT));
    c2_1.store_at(c_micropanel.add(2 * m + LANE_COUNT));
    c3_1.store_at(c_micropanel.add(3 * m + LANE_COUNT));
    c4_1.store_at(c_micropanel.add(4 * m + LANE_COUNT));
    c5_1.store_at(c_micropanel.add(5 * m + LANE_COUNT));
    c6_1.store_at(c_micropanel.add(6 * m + LANE_COUNT));
    c7_1.store_at(c_micropanel.add(7 * m + LANE_COUNT));

    c0_2.store_at(c_micropanel.add(LANE_COUNT * 2));
    c1_2.store_at(c_micropanel.add(m + LANE_COUNT * 2));
    c2_2.store_at(c_micropanel.add(2 * m + LANE_COUNT * 2));
    c3_2.store_at(c_micropanel.add(3 * m + LANE_COUNT * 2));
    c4_2.store_at(c_micropanel.add(4 * m + LANE_COUNT * 2));
    c5_2.store_at(c_micropanel.add(5 * m + LANE_COUNT * 2));
    c6_2.store_at(c_micropanel.add(6 * m + LANE_COUNT * 2));
    c7_2.store_at(c_micropanel.add(7 * m + LANE_COUNT * 2));
}

/// AVX2 microkernel for 32×8 matrix multiplication (experimental).
///
/// Maximum-size microkernel processing 32 rows of matrix A (4×8 AVX2 vectors) with 8 columns
/// of matrix B. Uses all 32 AVX2 registers for C accumulation, providing maximum register
/// utilization but likely causing register spilling.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (32×KC elements, stored as four 8×KC blocks)
/// * `b_panel` - Packed B matrix panel (KC×8 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 32×8 elements)
/// * `mr` - Actual number of rows in A block (≤ 32)
/// * `nr` - Actual number of columns in B block (≤ 8)
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Warning
/// **POTENTIAL BUG**: This kernel accesses `a_panel.data[k][N*LANE_COUNT..]` but APanel.data[k]
/// is only [f32; MR=8], not [f32; 32]. This will cause buffer overruns.
/// The APanel layout needs to be redesigned for kernels larger than 8×8.
///
/// # Performance Notes
/// High register pressure likely causes spilling. Benchmark carefully against smaller kernels.
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 32×8 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
/// - **WARNING**: Current implementation has serious memory access bugs
pub(crate) unsafe fn kernel_32x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0_0 = load_c(c_micropanel, mr);
    let mut c1_0 = load_c(c_micropanel.add(m), mr);
    let mut c2_0 = load_c(c_micropanel.add(2 * m), mr);
    let mut c3_0 = load_c(c_micropanel.add(3 * m), mr);
    let mut c4_0 = load_c(c_micropanel.add(4 * m), mr);
    let mut c5_0 = load_c(c_micropanel.add(5 * m), mr);
    let mut c6_0 = load_c(c_micropanel.add(6 * m), mr);
    let mut c7_0 = load_c(c_micropanel.add(7 * m), mr);

    let mut c0_1 = load_c(c_micropanel.add(LANE_COUNT), mr);
    let mut c1_1 = load_c(c_micropanel.add(m + LANE_COUNT), mr);
    let mut c2_1 = load_c(c_micropanel.add(2 * m + LANE_COUNT), mr);
    let mut c3_1 = load_c(c_micropanel.add(3 * m + LANE_COUNT), mr);
    let mut c4_1 = load_c(c_micropanel.add(4 * m + LANE_COUNT), mr);
    let mut c5_1 = load_c(c_micropanel.add(5 * m + LANE_COUNT), mr);
    let mut c6_1 = load_c(c_micropanel.add(6 * m + LANE_COUNT), mr);
    let mut c7_1 = load_c(c_micropanel.add(7 * m + LANE_COUNT), mr);

    let mut c0_2 = load_c(c_micropanel.add(LANE_COUNT * 2), mr);
    let mut c1_2 = load_c(c_micropanel.add(m + LANE_COUNT * 2), mr);
    let mut c2_2 = load_c(c_micropanel.add(2 * m + LANE_COUNT * 2), mr);
    let mut c3_2 = load_c(c_micropanel.add(3 * m + LANE_COUNT * 2), mr);
    let mut c4_2 = load_c(c_micropanel.add(4 * m + LANE_COUNT * 2), mr);
    let mut c5_2 = load_c(c_micropanel.add(5 * m + LANE_COUNT * 2), mr);
    let mut c6_2 = load_c(c_micropanel.add(6 * m + LANE_COUNT * 2), mr);
    let mut c7_2 = load_c(c_micropanel.add(7 * m + LANE_COUNT * 2), mr);

    let mut c0_3 = load_c(c_micropanel.add(LANE_COUNT * 3), mr);
    let mut c1_3 = load_c(c_micropanel.add(m + LANE_COUNT * 3), mr);
    let mut c2_3 = load_c(c_micropanel.add(2 * m + LANE_COUNT * 3), mr);
    let mut c3_3 = load_c(c_micropanel.add(3 * m + LANE_COUNT * 3), mr);
    let mut c4_3 = load_c(c_micropanel.add(4 * m + LANE_COUNT * 3), mr);
    let mut c5_3 = load_c(c_micropanel.add(5 * m + LANE_COUNT * 3), mr);
    let mut c6_3 = load_c(c_micropanel.add(6 * m + LANE_COUNT * 3), mr);
    let mut c7_3 = load_c(c_micropanel.add(7 * m + LANE_COUNT * 3), mr);

    for k in 0..kc {
        // 0

        let a_micropanel_0 = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Use the same interleaved broadcast pattern as simd_outer_product
        let b_lower_lane = b_micropanel.permute2f128::<0x00>();
        let b_upper_lane = b_micropanel.permute2f128::<0x11>();

        // First batch: broadcast elements 0 and 4, then 1 and 5
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        c0_0 = c0_0.fma(a_micropanel_0, b0_broadcast);
        c4_0 = c4_0.fma(a_micropanel_0, b4_broadcast);
        c1_0 = c1_0.fma(a_micropanel_0, b1_broadcast);
        c5_0 = c5_0.fma(a_micropanel_0, b5_broadcast);

        let b2_broadcast = b_lower_lane.permute::<0xAA>();
        let b6_broadcast = b_upper_lane.permute::<0xAA>();
        let b3_broadcast = b_lower_lane.permute::<0xFF>();
        let b7_broadcast = b_upper_lane.permute::<0xFF>();

        c2_0 = c2_0.fma(a_micropanel_0, b2_broadcast);
        c6_0 = c6_0.fma(a_micropanel_0, b6_broadcast);
        c3_0 = c3_0.fma(a_micropanel_0, b3_broadcast);
        c7_0 = c7_0.fma(a_micropanel_0, b7_broadcast);

        // 1

        let a_micropanel_1 = F32x8::load_aligned(a_panel.data[k][LANE_COUNT..].as_ptr());

        c0_1 = c0_1.fma(a_micropanel_1, b0_broadcast);
        c4_1 = c4_1.fma(a_micropanel_1, b4_broadcast);
        c1_1 = c1_1.fma(a_micropanel_1, b1_broadcast);
        c5_1 = c5_1.fma(a_micropanel_1, b5_broadcast);

        c2_1 = c2_1.fma(a_micropanel_1, b2_broadcast);
        c6_1 = c6_1.fma(a_micropanel_1, b6_broadcast);
        c3_1 = c3_1.fma(a_micropanel_1, b3_broadcast);
        c7_1 = c7_1.fma(a_micropanel_1, b7_broadcast);

        // 2

        let a_micropanel_2 = F32x8::load_aligned(a_panel.data[k][2 * LANE_COUNT..].as_ptr());

        c0_2 = c0_2.fma(a_micropanel_2, b0_broadcast);
        c4_2 = c4_2.fma(a_micropanel_2, b4_broadcast);
        c1_2 = c1_2.fma(a_micropanel_2, b1_broadcast);
        c5_2 = c5_2.fma(a_micropanel_2, b5_broadcast);

        c2_2 = c2_2.fma(a_micropanel_2, b2_broadcast);
        c6_2 = c6_2.fma(a_micropanel_2, b6_broadcast);
        c3_2 = c3_2.fma(a_micropanel_2, b3_broadcast);
        c7_2 = c7_2.fma(a_micropanel_2, b7_broadcast);

        //3

        let a_micropanel_3 = F32x8::load_aligned(a_panel.data[k][3 * LANE_COUNT..].as_ptr());

        c0_3 = c0_3.fma(a_micropanel_3, b0_broadcast);
        c4_3 = c4_3.fma(a_micropanel_3, b4_broadcast);
        c1_3 = c1_3.fma(a_micropanel_3, b1_broadcast);
        c5_3 = c5_3.fma(a_micropanel_3, b5_broadcast);

        c2_3 = c2_3.fma(a_micropanel_3, b2_broadcast);
        c6_3 = c6_3.fma(a_micropanel_3, b6_broadcast);
        c3_3 = c3_3.fma(a_micropanel_3, b3_broadcast);
        c7_3 = c7_3.fma(a_micropanel_3, b7_broadcast);
    }

    c0_0.store_at(c_micropanel);
    c1_0.store_at(c_micropanel.add(m));
    c2_0.store_at(c_micropanel.add(2 * m));
    c3_0.store_at(c_micropanel.add(3 * m));
    c4_0.store_at(c_micropanel.add(4 * m));
    c5_0.store_at(c_micropanel.add(5 * m));
    c6_0.store_at(c_micropanel.add(6 * m));
    c7_0.store_at(c_micropanel.add(7 * m));

    c0_1.store_at(c_micropanel.add(LANE_COUNT));
    c1_1.store_at(c_micropanel.add(m + LANE_COUNT));
    c2_1.store_at(c_micropanel.add(2 * m + LANE_COUNT));
    c3_1.store_at(c_micropanel.add(3 * m + LANE_COUNT));
    c4_1.store_at(c_micropanel.add(4 * m + LANE_COUNT));
    c5_1.store_at(c_micropanel.add(5 * m + LANE_COUNT));
    c6_1.store_at(c_micropanel.add(6 * m + LANE_COUNT));
    c7_1.store_at(c_micropanel.add(7 * m + LANE_COUNT));

    c0_2.store_at(c_micropanel.add(LANE_COUNT * 2));
    c1_2.store_at(c_micropanel.add(m + LANE_COUNT * 2));
    c2_2.store_at(c_micropanel.add(2 * m + LANE_COUNT * 2));
    c3_2.store_at(c_micropanel.add(3 * m + LANE_COUNT * 2));
    c4_2.store_at(c_micropanel.add(4 * m + LANE_COUNT * 2));
    c5_2.store_at(c_micropanel.add(5 * m + LANE_COUNT * 2));
    c6_2.store_at(c_micropanel.add(6 * m + LANE_COUNT * 2));
    c7_2.store_at(c_micropanel.add(7 * m + LANE_COUNT * 2));

    c0_3.store_at(c_micropanel.add(LANE_COUNT * 3));
    c1_3.store_at(c_micropanel.add(m + LANE_COUNT * 3));
    c2_3.store_at(c_micropanel.add(2 * m + LANE_COUNT * 3));
    c3_3.store_at(c_micropanel.add(3 * m + LANE_COUNT * 3));
    c4_3.store_at(c_micropanel.add(4 * m + LANE_COUNT * 3));
    c5_3.store_at(c_micropanel.add(5 * m + LANE_COUNT * 3));
    c6_3.store_at(c_micropanel.add(6 * m + LANE_COUNT * 3));
    c7_3.store_at(c_micropanel.add(7 * m + LANE_COUNT * 3));
}

/// AVX2 microkernel for 8×4 matrix multiplication.
///
/// Specialized microkernel for cases where the B matrix block has only 4 columns.
/// Uses B-element broadcasting where individual B elements are broadcast to all lanes
/// and multiplied with the A vector. This is more efficient than the general 8×8 kernel
/// when nr ≤ 4.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (8×KC elements)
/// * `b_panel` - Packed B matrix panel (KC×4 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 8×4 elements)
/// * `mr` - Actual number of rows in A block (≤ 8)
/// * `nr` - Actual number of columns in B block (≤ 4)
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 8×4 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
pub(crate) unsafe fn kernel_8x4(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = load_c(c_micropanel, mr);
    let mut c1 = load_c(c_micropanel.add(m), mr);
    let mut c2 = load_c(c_micropanel.add(2 * m), mr);
    let mut c3 = load_c(c_micropanel.add(3 * m), mr);

    for k in 0..kc {
        let a_micropanel = F32x8::load_aligned(a_panel.data[k].as_ptr());

        let mut b_micropanel = F32x8::broadcast(&b_panel.data[k][0]);
        c0 = c0.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][1]);
        c1 = c1.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][2]);
        c2 = c2.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][3]);
        c3 = c3.fma(a_micropanel, b_micropanel);
    }

    c0.store_at(c_micropanel);
    c1.store_at(c_micropanel.add(m));
    c2.store_at(c_micropanel.add(2 * m));
    c3.store_at(c_micropanel.add(3 * m));
}

/// AVX2 microkernel for 8×2 matrix multiplication.
///
/// Minimal microkernel for cases where the B matrix block has only 2 columns.
/// Uses B-element broadcasting for maximum efficiency with small column counts.
/// This kernel minimizes memory traffic by processing only the required columns.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (8×KC elements)
/// * `b_panel` - Packed B matrix panel (KC×2 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 8×2 elements)
/// * `mr` - Actual number of rows in A block (≤ 8)
/// * `nr` - Actual number of columns in B block (≤ 2)
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 8×2 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
pub(crate) unsafe fn kernel_8x2(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = load_c(c_micropanel, mr);
    let mut c1 = load_c(c_micropanel.add(m), mr);

    for k in 0..kc {
        let a_micropanel = F32x8::load_aligned(a_panel.data[k].as_ptr());

        let mut b_micropanel = F32x8::broadcast(&b_panel.data[k][0]);
        c0 = c0.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][1]);
        c1 = c1.fma(a_micropanel, b_micropanel);
    }

    c0.store_at(c_micropanel);
    c1.store_at(c_micropanel.add(m));
}

/// AVX2 microkernel for 8×6 matrix multiplication.
///
/// Microkernel for 6-column B matrix blocks. Uses B-element broadcasting to handle
/// the odd column count efficiently. This kernel bridges the gap between the 8×4
/// and 8×8 kernels when B has 5-6 columns.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (8×KC elements)
/// * `b_panel` - Packed B matrix panel (KC×6 elements)
/// * `c_micropanel` - Output C matrix block (column-major, 8×6 elements)
/// * `mr` - Actual number of rows in A block (≤ 8)
/// * `nr` - Actual number of columns in B block (≤ 6)
/// * `kc` - Inner dimension
/// * `m` - Leading dimension of matrix C
///
/// # Safety
/// - `c_micropanel` must point to valid memory for at least 8×6 f32 elements
/// - Matrix panels must contain valid data for the specified dimensions
pub(crate) unsafe fn kernel_8x6(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = load_c(c_micropanel, mr);
    let mut c1 = load_c(c_micropanel.add(m), mr);
    let mut c2 = load_c(c_micropanel.add(2 * m), mr);
    let mut c3 = load_c(c_micropanel.add(3 * m), mr);
    let mut c4 = load_c(c_micropanel.add(4 * m), mr);
    let mut c5 = load_c(c_micropanel.add(5 * m), mr);

    for k in 0..kc {
        let a_micropanel = F32x8::load_aligned(a_panel.data[k].as_ptr());

        let mut b_micropanel = F32x8::broadcast(&b_panel.data[k][0]);
        c0 = c0.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][1]);
        c1 = c1.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][2]);
        c2 = c2.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][3]);
        c3 = c3.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][4]);
        c4 = c4.fma(a_micropanel, b_micropanel);

        b_micropanel = F32x8::broadcast(&b_panel.data[k][5]);
        c5 = c5.fma(a_micropanel, b_micropanel);
    }

    c0.store_at(c_micropanel);
    c1.store_at(c_micropanel.add(m));
    c2.store_at(c_micropanel.add(2 * m));
    c3.store_at(c_micropanel.add(3 * m));
    c4.store_at(c_micropanel.add(4 * m));
    c5.store_at(c_micropanel.add(5 * m));
}
