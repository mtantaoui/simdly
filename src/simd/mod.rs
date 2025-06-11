#[cfg(avx2)]
pub mod avx2;

#[cfg(neon)]
pub mod neon;

pub mod traits;
