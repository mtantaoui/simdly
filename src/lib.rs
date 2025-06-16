#![cfg_attr(
    rustc_channel = "nightly",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

pub mod simd;
