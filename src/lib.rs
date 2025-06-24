#![cfg_attr(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        rustc_channel = "nightly"
    ),
    feature(avx512_target_feature, stdarch_x86_avx512)
)]

pub mod simd;

pub const MR: usize = 8;
pub const NR: usize = 6;

// pub const MC: usize = MR * 4;
// pub const NC: usize = NR * 8;
// pub const KC: usize = 512;

pub const NC: usize = NR * 12;
pub const KC: usize = 256;
pub const MC: usize = MR * 48;
