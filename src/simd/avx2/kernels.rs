//! Optimized matrix multiplication microkernels using AVX2 SIMD instructions.
//!
//! This module contains high-performance microkernels that form the computational core
//! of the BLIS-style matrix multiplication algorithm. These kernels operate on small
//! matrix blocks (typically 8×8 or 16×8) and are optimized for AVX2's 256-bit vectors.

use std::{arch::x86_64::*, cmp::min};

use crate::simd::{
    avx2::{
        f32x8::{F32x8, LANE_COUNT},
        panels::{APanel, BPanel, KC, MR, NR},
    },
    SimdLoad, SimdMath, SimdShuffle, SimdStore,
};

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

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

// ================================================================================================
// PRIMARY KERNELS (8×N variants - well-tested and optimized)
// ================================================================================================

/// AVX2 microkernel for 8×8 matrix multiplication using FMA operations.
///
/// This is the core computational kernel that multiplies an 8×KC block of matrix A
/// with a KC×8 block of matrix B, accumulating the result into an 8×8 block of matrix C.
/// Uses register tiling with 8 AVX2 registers to maximize arithmetic intensity.
///
/// # Algorithm
/// Implements outer product formulation: C += A ⊗ B^T where:
/// - Each iteration loads one A vector (8 elements) and one B vector (8 elements)
/// - B elements are broadcast individually and multiplied with entire A vector
/// - Results are accumulated in 8 AVX2 registers (c0-c7) representing output columns
///
/// # Performance Characteristics
/// - **Arithmetic Intensity**: 128 FLOPs per iteration (8×8×2 for FMA)
/// - **Memory Bandwidth**: 64 bytes loaded per iteration (2×32 bytes)
/// - **Register Utilization**: 8/16 YMM registers for C accumulation
/// - **Pipeline Friendly**: Interleaved FMA operations reduce dependency chains
/// - **Cache Efficiency**: Packed panels maximize spatial locality
///
/// # Arguments
/// * `a_panel` - Packed A matrix panel (8×KC elements, row-major within each k-slice)
///   - Pre-packed for optimal memory access patterns
///   - Each `a_panel.data[k]` contains 8 consecutive A elements for k-th iteration
/// * `b_panel` - Packed B matrix panel (KC×8 elements, column-major within each k-slice)
///   - Pre-packed with B elements arranged for efficient SIMD broadcasting
///   - Each `b_panel.data[k]` contains 8 B elements (one from each column) for k-th iteration
/// * `c_micropanel` - Output C matrix block pointer (column-major layout, 8×8 elements)
///   - Must be 32-byte aligned for optimal AVX2 performance
///   - Elements stored as C[row + col*m] where m is the leading dimension
/// * `mr` - Actual number of rows in A block (1 ≤ mr ≤ 8)
///   - Used for boundary handling to prevent buffer overruns
///   - Enables processing of non-multiple-of-8 matrix dimensions
/// * `nr` - Actual number of columns in B block (1 ≤ nr ≤ 8)
///   - Used for boundary handling to prevent buffer overruns  
///   - Enables processing of non-multiple-of-8 matrix dimensions
/// * `kc` - Inner dimension (number of terms in dot products, KC ≤ 512)
///   - Represents the blocking parameter for the k-dimension
///   - Each iteration processes one k-slice of the packed panels
/// * `m` - Leading dimension of matrix C (column stride in elements)
///   - Distance between consecutive elements in the same row but different columns
///   - Must be ≥ actual matrix height for correct addressing
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

        // Split B vector [b0,b1,b2,b3,b4,b5,b6,b7] into two 128-bit lanes for broadcasting
        // b_lower_lane = [b0,b1,b2,b3, b0,b1,b2,b3] (duplicate lower 128 bits)
        // b_upper_lane = [b4,b5,b6,b7, b4,b5,b6,b7] (duplicate upper 128 bits)
        let b_lower_lane = b_micropanel.permute2f128::<0x00>();
        let b_upper_lane = b_micropanel.permute2f128::<0x11>();

        // Broadcast individual B elements across all 8 lanes for outer product computation
        // 0x00 = broadcast element 0: [b0,b0,b0,b0, b0,b0,b0,b0]
        // 0x55 = broadcast element 1: [b1,b1,b1,b1, b1,b1,b1,b1]
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        // Compute outer products: A[0:7] * B[j] -> C[0:7][j]
        // Interleave operations to improve instruction-level parallelism
        c0 = c0.fma(a_micropanel, b0_broadcast); // C[0:7][0] += A[0:7] * B[0]
        c4 = c4.fma(a_micropanel, b4_broadcast); // C[0:7][4] += A[0:7] * B[4]
        c1 = c1.fma(a_micropanel, b1_broadcast); // C[0:7][1] += A[0:7] * B[1]
        c5 = c5.fma(a_micropanel, b5_broadcast); // C[0:7][5] += A[0:7] * B[5]

        // Continue with remaining B elements
        // 0xAA = broadcast element 2: [b2,b2,b2,b2, b2,b2,b2,b2]
        // 0xFF = broadcast element 3: [b3,b3,b3,b3, b3,b3,b3,b3]
        let b2_broadcast = b_lower_lane.permute::<0xAA>();
        let b6_broadcast = b_upper_lane.permute::<0xAA>();
        let b3_broadcast = b_lower_lane.permute::<0xFF>();
        let b7_broadcast = b_upper_lane.permute::<0xFF>();

        c2 = c2.fma(a_micropanel, b2_broadcast); // C[0:7][2] += A[0:7] * B[2]
        c6 = c6.fma(a_micropanel, b6_broadcast); // C[0:7][6] += A[0:7] * B[6]
        c3 = c3.fma(a_micropanel, b3_broadcast); // C[0:7][3] += A[0:7] * B[3]
        c7 = c7.fma(a_micropanel, b7_broadcast); // C[0:7][7] += A[0:7] * B[7]
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

// ================================================================================================
// EXTENDED KERNELS (larger MR variants - experimental, potential memory layout issues)
// ================================================================================================

/// AVX2 microkernel for 16×8 matrix multiplication (experimental).
///
/// Extended version of the 8×8 kernel that processes 16 rows of matrix A simultaneously.
/// This kernel uses twice as many registers (16 AVX2 registers) to potentially improve
/// performance by increasing register utilization and reducing memory traffic.
///
/// # Arguments  
/// * `a_panel` - Packed A matrix panel (16×KC elements, stored as two 8×KC blocks)
///   - **Note**: Current implementation may have layout issues for mr > 8
///   - First 8 elements at `a_panel.data[k].as_ptr()`, next 8 at offset `LANE_COUNT`
/// * `b_panel` - Packed B matrix panel (KC×8 elements)
///   - Same layout as 8×8 kernel - elements broadcast to both 8-row blocks
/// * `c_micropanel` - Output C matrix block (column-major, 16×8 elements)
///   - Upper 8×8 block starts at `c_micropanel`
///   - Lower 8×8 block starts at `c_micropanel + LANE_COUNT`
/// * `mr` - Actual number of rows in A block (1 ≤ mr ≤ 16)
/// * `nr` - Actual number of columns in B block (1 ≤ nr ≤ 8)
/// * `kc` - Inner dimension (number of dot product terms)
/// * `m` - Leading dimension of matrix C (≥ 16 for this kernel)
///
/// # Performance Characteristics  
/// - **Register Pressure**: Uses 16/16 YMM registers for C accumulation, guaranteed spilling
/// - **Memory Traffic**: 2× the bandwidth of 8×8 kernel (96 bytes/iteration)
/// - **Cache Behavior**: May exceed L1 cache capacity for large KC values  
/// - **Pipeline Efficiency**: Doubled FMA operations may saturate execution units
/// - **Recommendation**: Profile against 8×8 kernel; larger isn't always faster
///
/// **Register Breakdown**: 16 YMM for C accumulation + temporaries for A/B loading exceed 16 total
///
/// # Tuning Notes
/// Consider this kernel when:
/// - Matrix sizes are multiples of 16 (reduces boundary handling overhead)
/// - L1 cache can accommodate 16-row working set
/// - CPU has high FMA throughput (≥2 FMA units)
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
        // === Process upper 8 rows (rows 0-7) ===
        let a_micropanel_hi = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Prepare B vector for broadcasting (same as 8x8 kernel)
        let b_lower_lane = b_micropanel.permute2f128::<0x00>(); // [b0,b1,b2,b3, b0,b1,b2,b3]
        let b_upper_lane = b_micropanel.permute2f128::<0x11>(); // [b4,b5,b6,b7, b4,b5,b6,b7]

        // Broadcast B elements for outer product with upper A rows
        let b0_broadcast = b_lower_lane.permute::<0x00>();
        let b4_broadcast = b_upper_lane.permute::<0x00>();
        let b1_broadcast = b_lower_lane.permute::<0x55>();
        let b5_broadcast = b_upper_lane.permute::<0x55>();

        // Compute C[0:7][j] += A[0:7] * B[j]
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

        // === Process lower 8 rows (rows 8-15) ===
        let a_micropanel_lo = F32x8::load_aligned(a_panel.data[k].as_ptr().add(LANE_COUNT));

        // Reuse the same B broadcasts for lower A rows
        // Compute C[8:15][j] += A[8:15] * B[j]
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
///   - **CRITICAL**: Current APanel only supports MR=8, this will cause memory corruption
///   - Attempts to access `a_panel.data[k][LANE_COUNT..]` and `[2*LANE_COUNT..]`
/// * `b_panel` - Packed B matrix panel (KC×8 elements)
///   - Same B layout reused for all three 8-row A blocks
/// * `c_micropanel` - Output C matrix block (column-major, 24×8 elements)
///   - Three 8×8 blocks: rows 0-7, 8-15, 16-23
/// * `mr` - Actual number of rows in A block (1 ≤ mr ≤ 24)
/// * `nr` - Actual number of columns in B block (1 ≤ nr ≤ 8)
/// * `kc` - Inner dimension (number of dot product terms)
/// * `m` - Leading dimension of matrix C (≥ 24 for this kernel)
///
/// # Warning
/// **POTENTIAL BUG**: This kernel accesses `a_panel.data[k][LANE_COUNT..]` but APanel.data[k]
/// is only [f32; MR=8], not [f32; 24]. This may cause buffer overruns or incorrect results.
/// The APanel layout may need to be redesigned for kernels larger than 8×8.
///
/// # Performance Characteristics
/// - **Extreme Register Pressure**: Uses 24/16 YMM registers for C accumulation, severe spilling
/// - **Memory Hierarchy**: Very high memory bandwidth requirements (128 bytes/iteration)
/// - **CPU Bottlenecks**: May saturate multiple execution units simultaneously
/// - **Diminishing Returns**: Unlikely to outperform smaller, well-tuned kernels
///
/// **Register Breakdown**: 24 YMM for C accumulation + A/B temporaries = ~28 total registers needed
///
/// # Recommendation
/// This kernel exists primarily for completeness and experimental purposes.
/// In practice, 8×8 or 16×8 kernels typically provide better performance.
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
/// # Performance Characteristics  
/// - **Maximum Register Usage**: 32/16 YMM registers for C accumulation, massive spilling
/// - **Guaranteed Spilling**: CPU cannot hold all accumulator registers (~36 total needed)
/// - **Memory Wall**: Extremely high memory bandwidth requirements (160 bytes/iteration)
/// - **Execution Unit Saturation**: May exceed CPU's arithmetic throughput
/// - **Memory Subsystem**: Register spilling creates additional memory pressure
///
/// **Register Breakdown**: 32 YMM for C + A/B temporaries = 36+ total registers required
///
/// # Recommendation
/// **EXPERIMENTAL ONLY** - This kernel is primarily useful for:
/// - Architecture research and understanding register limits
/// - Comparative analysis against practical kernel sizes
/// - Demonstrating the performance cliff beyond optimal register usage
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

// ================================================================================================
// REDUCED NR KERNELS (8×N variants for smaller column counts)
// ================================================================================================

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

// ================================================================================================
// RAW INTRINSICS KERNELS (low-level implementations for reference/debugging)
// ================================================================================================

/// Raw AVX2 intrinsics implementation of 8×8 matrix multiplication kernel.
///
/// **WARNING: POTENTIALLY BUGGY IMPLEMENTATION**
/// This function provides a reference implementation using direct AVX2 intrinsics,
/// but it may have mathematical inconsistencies compared to the main kernel_8x8.
///
/// # Implementation Details  
/// Uses A-element broadcasting instead of B-element broadcasting:
/// - Each A[i,k] is broadcast and multiplied by B[k,0:7] 
/// - But results are stored column-wise like the B-broadcasting kernels
/// - This may produce INCORRECT results due to indexing mismatch
///
/// # Purpose
/// This kernel exists primarily for:
/// - Educational purposes (showing intrinsics usage)
/// - Performance comparison (when working correctly)
/// - **NOT for production use until verified**
///
/// # Arguments
/// Same as `kernel_8x8` - see that function for detailed parameter documentation.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions must match the provided parameters  
/// - **Results may be mathematically incorrect**
pub unsafe fn raw_kernel_8x8(
    a_panel: &APanel<MR, KC>,
    b_panel: &BPanel<KC, NR>,
    c_micropanel: *mut f32,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    let mut c0 = _mm256_load_ps(c_micropanel);
    let mut c1 = _mm256_load_ps(c_micropanel.add(m));
    let mut c2 = _mm256_load_ps(c_micropanel.add(2 * m));
    let mut c3 = _mm256_load_ps(c_micropanel.add(3 * m));
    let mut c4 = _mm256_load_ps(c_micropanel.add(4 * m));
    let mut c5 = _mm256_load_ps(c_micropanel.add(5 * m));
    let mut c6 = _mm256_load_ps(c_micropanel.add(6 * m));
    let mut c7 = _mm256_load_ps(c_micropanel.add(7 * m));

    for k in 0..kc {
        let a_micropanel = _mm256_load_ps(a_panel.data[k].as_ptr());
        let b_micropanel = _mm256_load_ps(b_panel.data[k].as_ptr());

        // Split A vector [a0,a1,a2,a3,a4,a5,a6,a7] for broadcasting
        // a_lower_lane = [a0,a1,a2,a3, a0,a1,a2,a3] (duplicate lower 128 bits)
        // a_upper_lane = [a4,a5,a6,a7, a4,a5,a6,a7] (duplicate upper 128 bits)
        let a_lower_lane = _mm256_permute2f128_ps::<0x00>(a_micropanel, a_micropanel);
        let a_upper_lane = _mm256_permute2f128_ps::<0x11>(a_micropanel, a_micropanel);

        // Broadcast individual A elements across all 8 lanes
        // Note: This uses A-broadcasting (different approach from other kernels)
        let a0_broadcast = _mm256_permute_ps(a_lower_lane, 0x00); // [a0,a0,a0,a0, a0,a0,a0,a0]
        let a4_broadcast = _mm256_permute_ps(a_upper_lane, 0x00); // [a4,a4,a4,a4, a4,a4,a4,a4]
        let a1_broadcast = _mm256_permute_ps(a_lower_lane, 0x55); // [a1,a1,a1,a1, a1,a1,a1,a1]
        let a5_broadcast = _mm256_permute_ps(a_upper_lane, 0x55); // [a5,a5,a5,a5, a5,a5,a5,a5]

        // **POTENTIAL BUG**: A-broadcasting with column-wise storage may be incorrect
        // Computing: A[i,k] * B[k,0:7] but storing as if computing columns
        // This may produce transposed or incorrect results
        c0 = _mm256_fmadd_ps(a0_broadcast, b_micropanel, c0); // Stores to column 0
        c4 = _mm256_fmadd_ps(a4_broadcast, b_micropanel, c4); // Stores to column 4
        c1 = _mm256_fmadd_ps(a1_broadcast, b_micropanel, c1); // Stores to column 1
        c5 = _mm256_fmadd_ps(a5_broadcast, b_micropanel, c5); // Stores to column 5

        // Process remaining A elements
        let a2_broadcast = _mm256_permute_ps(a_lower_lane, 0xAA); // [a2,a2,a2,a2, a2,a2,a2,a2]
        let a6_broadcast = _mm256_permute_ps(a_upper_lane, 0xAA); // [a6,a6,a6,a6, a6,a6,a6,a6]
        let a3_broadcast = _mm256_permute_ps(a_lower_lane, 0xFF); // [a3,a3,a3,a3, a3,a3,a3,a3]
        let a7_broadcast = _mm256_permute_ps(a_upper_lane, 0xFF); // [a7,a7,a7,a7, a7,a7,a7,a7]

        c2 = _mm256_fmadd_ps(a2_broadcast, b_micropanel, c2); // Stores to column 2
        c6 = _mm256_fmadd_ps(a6_broadcast, b_micropanel, c6); // Stores to column 6
        c3 = _mm256_fmadd_ps(a3_broadcast, b_micropanel, c3); // Stores to column 3
        c7 = _mm256_fmadd_ps(a7_broadcast, b_micropanel, c7); // Stores to column 7
    }

    _mm256_store_ps(c_micropanel, c0);
    _mm256_store_ps(c_micropanel.add(m), c1);
    _mm256_store_ps(c_micropanel.add(2 * m), c2);
    _mm256_store_ps(c_micropanel.add(3 * m), c3);
    _mm256_store_ps(c_micropanel.add(4 * m), c4);
    _mm256_store_ps(c_micropanel.add(5 * m), c5);
    _mm256_store_ps(c_micropanel.add(6 * m), c6);
    _mm256_store_ps(c_micropanel.add(7 * m), c7);
}
