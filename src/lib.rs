#![cfg_attr(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        rustc_channel = "nightly"
    ),
    feature(avx512_target_feature, stdarch_x86_avx512)
)]

pub mod simd;

pub const MR: usize = 16;
pub const NR: usize = 4;

pub const MC: usize = MR * 8;
pub const NC: usize = NR * 32;
pub const KC: usize = 1024;
