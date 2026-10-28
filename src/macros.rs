//! Architecture-selection macros.
//!
//! These, and the feature checks in `detect.rs`, are the only places where
//! architecture `cfg` predicates are written. Code that only exists on one
//! architecture uses the matching macro instead:
//!
//! | Macro        | Active when                            | Backends compiled         |
//! |--------------|----------------------------------------|---------------------------|
//! | `cfg_x86!`   | `x86_64` with SSE2 (not soft-float)    | AVX2, AVX-512 (runtime-selected) |
//! | `cfg_neon!`  | `aarch64` + `neon`                     | NEON                      |
//! | `cfg_simd!`  | either of the above                    |                           |
//!
//! The scalar backend is always compiled, as the fallback.
//!
//! `cfg_x86!` and `cfg_neon!` take items, or an expression:
//!
//! ```rust,ignore
//! cfg_x86! {
//!     mod avx2;
//! }
//!
//! let fast = cfg_x86!(if { has_avx2_fma() } else { false });
//! ```
//!
//! Unlike `if cfg!(..)`, the branch that is not selected is never compiled,
//! so it may name items that only exist on the other architecture.
//!
//! `dispatch!` is the entry point of every public function: it calls the
//! fastest available backend, see its docs.

macro_rules! cfg_x86 {
    (if { $($then:tt)* } else { $($else:tt)* }) => {{
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        let value = { $($then)* };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
        let value = { $($else)* };
        value
    }};
    ($($item:item)*) => {
        $(
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            $item
        )*
    };
}

macro_rules! cfg_neon {
    (if { $($then:tt)* } else { $($else:tt)* }) => {{
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        let value = { $($then)* };
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        let value = { $($else)* };
        value
    }};
    ($($item:item)*) => {
        $(
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            $item
        )*
    };
}

macro_rules! cfg_simd {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                all(target_arch = "x86_64", target_feature = "sse2"),
                all(target_arch = "aarch64", target_feature = "neon"),
            ))]
            $item
        )*
    };
}

/// Calls the fastest available backend: AVX-512, AVX2, NEON, then scalar.
///
/// Every arm but `scalar` is optional, and an arm is only compiled on its
/// architecture, so it may name items that only exist there:
///
/// ```rust,ignore
/// pub fn exp(x: &mut [f32]) {
///     dispatch! {
///         avx2 => avx2::exp(x),
///         neon => neon::exp(x),
///         scalar => scalar::exp(x),
///     }
/// }
/// ```
///
/// x86 arms run in an `unsafe` block once their features are detected, so
/// they can call `#[target_feature]` functions directly. An arm must need
/// nothing beyond its backend's features (see `detect::Backend`).
macro_rules! dispatch {
    (
        $(avx512 => $avx512:expr,)?
        $(avx2 => $avx2:expr,)?
        $(neon => $neon:expr,)?
        scalar => $scalar:expr $(,)?
    ) => {
        {
            #[allow(unused_labels, reason = "only the scalar arm may be compiled")]
            let value = 'dispatch: {
                $(
                    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                    #[allow(unused_unsafe, reason = "safe when the features are enabled at compile time")]
                    if $crate::detect::Backend::Avx512.is_available() {
                        // SAFETY: AVX-512F was just detected.
                        break 'dispatch unsafe { $avx512 };
                    }
                )?
                $(
                    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                    #[allow(unused_unsafe, reason = "safe when the features are enabled at compile time")]
                    if $crate::detect::Backend::Avx2.is_available() {
                        // SAFETY: AVX2 and FMA were just detected.
                        break 'dispatch unsafe { $avx2 };
                    }
                )?
                $(
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    if $crate::detect::Backend::Neon.is_available() {
                        break 'dispatch $neon;
                    }
                )?
                $scalar
            };
            value
        }
    };
}
