//! Backend-selection macros.
//!
//! These are the only place in the crate where a backend's `cfg` predicate is
//! written. Every module that is specific to one backend wraps its items in
//! the matching macro instead of repeating the predicate by hand:
//!
//! ```rust,ignore
//! cfg_avx2! {
//!     mod avx2;
//! }
//! ```
//!
//! The four predicates are mutually exclusive and cover every target, so
//! exactly one backend is compiled in:
//!
//! | Macro          | Active when                                         |
//! |----------------|-----------------------------------------------------|
//! | `cfg_avx512!`  | `x86_64` + `avx512f`                                |
//! | `cfg_avx2!`    | `x86_64` + `avx2` + `fma`, without `avx512f`        |
//! | `cfg_neon!`    | `aarch64` + `neon`                                  |
//! | `cfg_scalar!`  | none of the above                                   |

macro_rules! cfg_avx512 {
    ($($item:item)*) => {
        $(
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            $item
        )*
    };
}

macro_rules! cfg_avx2 {
    ($($item:item)*) => {
        $(
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma",
                not(target_feature = "avx512f"),
            ))]
            $item
        )*
    };
}

macro_rules! cfg_neon {
    ($($item:item)*) => {
        $(
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            $item
        )*
    };
}

macro_rules! cfg_scalar {
    ($($item:item)*) => {
        $(
            #[cfg(not(any(
                all(target_arch = "x86_64", target_feature = "avx512f"),
                all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"),
                all(target_arch = "aarch64", target_feature = "neon"),
            )))]
            $item
        )*
    };
}

// Defining `BACKEND` once per backend turns the "exactly one backend" claim
// above into a compile-time check: two active predicates would be a duplicate
// definition, zero would leave it undefined.
cfg_avx512! {
    #[allow(dead_code)]
    pub(crate) const BACKEND: &str = "avx512";
}
cfg_avx2! {
    #[allow(dead_code)]
    pub(crate) const BACKEND: &str = "avx2";
}
cfg_neon! {
    #[allow(dead_code)]
    pub(crate) const BACKEND: &str = "neon";
}
cfg_scalar! {
    #[allow(dead_code)]
    pub(crate) const BACKEND: &str = "scalar";
}

#[cfg(test)]
mod tests {
    #[test]
    fn exactly_one_backend_is_selected() {
        assert!(["avx512", "avx2", "neon", "scalar"].contains(&super::BACKEND));
    }
}
