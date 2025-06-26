#![cfg_attr(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        rustc_channel = "nightly"
    ),
    feature(avx512_target_feature, stdarch_x86_avx512)
)]

pub mod simd;

//ubuntu
pub const MR: usize = 8;
pub const NR: usize = 8;

// ubuntu
pub const NC: usize = 1024;
pub const KC: usize = 256;
pub const MC: usize = 64;

// // linux server
// pub const MC: usize = MR * 4;
// pub const NC: usize = NR * 8;
// pub const KC: usize = 512;
