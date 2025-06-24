use std::cmp::min;

// --- Constants often derived from architecture specifics or tuning ---

// For f32 on AVX-512, a common choice for MR is around 12-16.
// NR is often a multiple of SIMD width (16 floats for __m512).
// Let's pick some representative values. These are the hardest to derive purely from cache sizes.
const TARGET_MR_F32: usize = 16; // Micro-kernel rows (e.g., for F32x16 SIMD type)
const TARGET_NR_F32: usize = 16; // Micro-kernel columns (e.g., 1 vector wide for F32x16)
                                 // Or NR could be 32 (2 vectors wide). Let's start with 16.

const BYTES_PER_ELEMENT_F32: usize = 4; // sizeof(f32)

// Fractions of cache to target. These are heuristics.
const L1D_UTILIZATION_FOR_B_PANEL: f64 = 0.5; // Leave room for A elements, stack, etc.
const L2_UTILIZATION_FOR_A_PACKED: f64 = 0.4; // Packed A panel in L2
const L3_UTILIZATION_FOR_B_PACKED: f64 = 0.3; // Packed B panel in L3 (C also competes)
const L3_UTILIZATION_FOR_C_PANEL: f64 = 0.3; // C panel in L3

#[derive(Debug, Clone, Copy)]
pub struct BlasParams {
    pub mr: usize,
    pub nr: usize,
    pub kc: usize, // L1-target for B micro-panel, also depth of A/B packed panels
    pub mc: usize, // L2-target for A packed panel height
    pub nc: usize, // L3-target for B packed panel width
}

/// Computes BLIS-like kernel parameters based on cache sizes for f32.
///
/// This function provides heuristic estimates. Optimal values often require
/// architecture-specific tuning and benchmarking.
///
/// # Arguments
/// * `l1d_cache_size_bytes`: L1 data cache size per core in bytes.
/// * `l2_cache_size_bytes`: L2 cache size per core in bytes.
/// * `l3_cache_size_bytes`: L3 cache size (total or per-tile/complex) in bytes.
///                          If shared, consider the portion available to a single core.
///
/// # Returns
/// A `BlasParams` struct with estimated MR, NR, MC, KC, NC.
pub fn calculate_f32_blas_params(
    l1d_cache_size_bytes: usize,
    l2_cache_size_bytes: usize,
    l3_cache_size_bytes: usize,
) -> BlasParams {
    // 1. Determine MR and NR (Micro-kernel dimensions - Register blocking)
    // These are heavily architecture-dependent and less directly derived from cache sizes alone.
    // We use predefined targets for f32 on an AVX-512 like system.
    // MR must be such that MR elements can be processed by the SIMD vector type used for A columns.
    // NR must be such that NR elements can be processed by SIMD vectors for B rows/C columns.
    let mr = TARGET_MR_F32; // e.g., 16 (matches F32x16 vector height)
    let nr = TARGET_NR_F32; // e.g., 16 (1 F32x16 vector wide)

    // --- Sanity checks for cache sizes ---
    if l1d_cache_size_bytes == 0 || l2_cache_size_bytes == 0 {
        // l3_cache_size_bytes can be 0 if no L3
        panic!("L1D and L2 cache sizes must be greater than 0.");
    }
    if mr == 0 || nr == 0 || BYTES_PER_ELEMENT_F32 == 0 {
        panic!("MR, NR, and element size must be greater than 0.");
    }

    // 2. Determine KC (L1 Cache blocking parameter)
    // Aim for a micro-panel of B (KC x NR) to fit in a fraction of L1d.
    // KC * NR * sizeof(f32) <= l1d_cache_size_bytes * L1D_UTILIZATION_FOR_B_PANEL
    let target_b_micropanel_l1_size =
        (l1d_cache_size_bytes as f64 * L1D_UTILIZATION_FOR_B_PANEL) as usize;

    let mut kc = target_b_micropanel_l1_size / (nr * BYTES_PER_ELEMENT_F32);

    // KC should generally not be excessively small. Often a multiple of MR or NR, or cache line size.
    // Let's ensure it's at least, say, MR or NR.
    kc = kc.max(mr).max(nr);
    // And often a multiple of some SIMD vector width or a common block size (e.g., 16, 32, 64).
    // For simplicity, we might align it to a multiple of, say, 16 or 32.
    // A common strategy is to make KC a reasonably large value that still fits L1 constraints.
    // BLIS KC for f32/zen4 can be 384 or 512.
    // Let's cap it by a reasonable heuristic if calculation is too large, or ensure it's not zero.
    kc = if kc == 0 { nr.max(mr) } else { kc }; // ensure not zero
                                                // Make KC a multiple of, say, 16 (SIMD vector width in elements) for cleaner packing/looping.
    kc = (kc / 16).max(1) * 16; // Ensure it's at least 16 and multiple of 16.
                                // Example: if l1d=32KB, L1D_UTIL=0.5 -> 16KB. 16384 / (16*4) = 16384 / 64 = 256.
                                // This is a common KC for doubles; for floats, it could be larger.
                                // If l1d=48KB, L1D_UTIL=0.5 -> 24KB. 24576 / 64 = 384.

    // 3. Determine MC (L2 Cache blocking parameter for A)
    // Aim for a packed panel of A (MC x KC) to fit in a fraction of L2.
    // MC * KC * sizeof(f32) <= l2_cache_size_bytes * L2_UTILIZATION_FOR_A_PACKED
    let target_a_packed_l2_size =
        (l2_cache_size_bytes as f64 * L2_UTILIZATION_FOR_A_PACKED) as usize;

    let mut mc = target_a_packed_l2_size / (kc * BYTES_PER_ELEMENT_F32);

    // MC must be a multiple of MR.
    mc = if mc == 0 { mr } else { (mc / mr).max(1) * mr }; // Ensure at least MR and multiple of MR.
                                                           // Typical BLIS MC for f32/zen4 can be 144 or higher.

    // 4. Determine NC (L3 Cache blocking parameter for B and C)
    // This is more complex. We need to consider fitting a packed B panel (KC x NC)
    // AND the corresponding C panel (MC x NC) in L3.
    // Let's prioritize B_packed fitting a portion of L3, and then check C.
    // KC * NC * sizeof(f32) approx L3_UTILIZATION_FOR_B_PACKED * l3_cache_size_bytes
    // MC * NC * sizeof(f32) approx L3_UTILIZATION_FOR_C_PANEL * l3_cache_size_bytes
    // Choose NC based on the tighter constraint.

    let mut nc_from_b_packed = 0;
    let mut nc_from_c_panel = 0;

    if l3_cache_size_bytes > 0 {
        let target_b_packed_l3_size =
            (l3_cache_size_bytes as f64 * L3_UTILIZATION_FOR_B_PACKED) as usize;
        nc_from_b_packed = target_b_packed_l3_size / (kc * BYTES_PER_ELEMENT_F32);

        let target_c_panel_l3_size =
            (l3_cache_size_bytes as f64 * L3_UTILIZATION_FOR_C_PANEL) as usize;
        nc_from_c_panel = target_c_panel_l3_size / (mc * BYTES_PER_ELEMENT_F32);
    }

    let mut nc = if l3_cache_size_bytes == 0 {
        // No L3, NC might be smaller, targeting L2 for B or just streaming.
        // For simplicity, let's make it some multiple of MC or a fixed large value for streaming.
        // This part is highly heuristic without L3.
        // Let's make it similar to MC, or a few times larger.
        (mc * 4).max(nr * 16) // Arbitrary large value if no L3, e.g. 256*NR
    } else {
        min(nc_from_b_packed, nc_from_c_panel)
    };

    // NC must be a multiple of NR.
    nc = if nc == 0 { nr } else { (nc / nr).max(1) * nr };
    // Typical BLIS NC for f32/zen4 can be very large (e.g., 4080, 8160),
    // implying B is well-streamed or L3 is large and shared effectively.
    // Our heuristic might be conservative.

    BlasParams { mr, nr, kc, mc, nc }
}

fn main() {
    // Example usage:
    // Values for a hypothetical CPU or a known one like Zen 4
    // Zen 4 EPYC 9R14-like (using your earlier lscpu output as rough guide):
    // L1d: 32 KiB per core (though some Zen 4 are 48KiB, let's use 32 based on prior context)
    // L2: 1 MiB per core
    // L3: 32 MiB per socket/CCD (let's assume a portion available, e.g., 4-8MB if it's per CCX/CCD of 8 cores)
    // For this heuristic, let's assume a single core has good access to a portion of L3.
    // If L3 is shared by many cores, effective L3 per core is smaller.
    let l1d_kib = 32;
    let l2_mib = 1;
    let l3_mib_effective_per_core = 4; // Assume 4MiB of the L3 is effectively usable by one core's stream

    let params = calculate_f32_blas_params(
        l1d_kib * 1024,
        l2_mib * 1024 * 1024,
        l3_mib_effective_per_core * 1024 * 1024,
    );
    println!("Calculated BLAS params for f32:");
    println!(
        "L1d: {} KiB, L2: {} MiB, L3_eff: {} MiB",
        l1d_kib, l2_mib, l3_mib_effective_per_core
    );
    println!("{:#?}", params);

    // Example: Typical Zen 4 (desktop/server grade) might be closer to:
    // L1d: 48KiB, L2: 1MiB, L3: 32MiB (per CCD of 8 cores, so ~4MiB/core effective for some workloads)
    let l1d_zen4_desktop = 48;
    let l2_zen4_desktop = 1;
    let l3_zen4_ccd_slice = 4; // Assuming one core targets a slice of a larger CCD L3
    let params_zen4_like = calculate_f32_blas_params(
        l1d_zen4_desktop * 1024,
        l2_zen4_desktop * 1024 * 1024,
        l3_zen4_ccd_slice * 1024 * 1024,
    );
    println!("\nCalculated BLAS params for f32 (Zen4-like desktop):");
    println!(
        "L1d: {} KiB, L2: {} MiB, L3_slice: {} MiB",
        l1d_zen4_desktop, l2_zen4_desktop, l3_zen4_ccd_slice
    );
    println!("{:#?}", params_zen4_like);

    // What BLIS actually uses for Zen 4 SGEMM:
    // MR=12, NR=32 (or MR=16, NR=16 for some variants)
    // KC (L1 target) = 384 or 512
    // MC (L2 target) = e.g., 144 (if MR=12) or could be higher
    // NC (L3 target) = e.g., 8160
    // Our heuristic might differ significantly, especially for NC.
}
