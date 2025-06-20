#![cfg_attr(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        rustc_channel = "nightly"
    ),
    feature(stdsimd, avx512_target_feature)
)]

pub mod simd;
