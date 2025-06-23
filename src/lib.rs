#![cfg_attr(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        rustc_channel = "nightly"
    ),
    feature(avx512_target_feature, stdarch_x86_avx512)
)]

pub mod simd;
