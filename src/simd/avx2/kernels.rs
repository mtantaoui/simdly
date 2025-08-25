use crate::simd::{
    avx2::{
        f32x8::{F32x8, LANE_COUNT},
        panels::{APanel, BPanel, KC, MR, NR},
    },
    SimdLoad, SimdMath, SimdShuffle, SimdStore,
};

/// Helper function to load C matrix elements, choosing between full and partial load
#[inline(always)]
unsafe fn load_c(ptr: *const f32, size: usize) -> F32x8 {
    if size == LANE_COUNT {
        F32x8::load(ptr, size)
    } else {
        F32x8::load_partial(ptr, size)
    }
}

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
        // let a_micropanel = F32x8::from(a_panel.data[k].as_slice());
        // let b_micropanel = F32x8::from(b_panel.data[k].as_slice());

        let a_micropanel = F32x8::load_aligned(a_panel.data[k].as_ptr());
        let b_micropanel = F32x8::load_aligned(b_panel.data[k].as_ptr());

        // Use the same interleaved broadcast pattern as simd_outer_product
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
