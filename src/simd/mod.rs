#[cfg(all(avx512, rustc_channel = "nightly"))]
pub mod avx512;

#[cfg(avx2)]
pub mod avx2;

#[cfg(neon)]
pub mod neon;

pub mod traits;
pub mod utils;
