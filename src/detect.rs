//! CPU feature detection for backend selection.
//!
//! With the `std` feature, x86_64 features are detected at runtime (and cached
//! by `std`); features enabled at compile time fold to `true`, so the check
//! disappears. Without `std`, only compile-time target features count.

#![allow(dead_code, reason = "used by the dispatchers once functions land")]

/// A SIMD backend, or the scalar fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl Backend {
    /// Whether this backend can run here.
    #[inline(always)]
    pub(crate) fn is_available(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Avx2 => cfg_x86!(if { has_avx2_fma() } else { false }),
            Self::Avx512 => cfg_x86!(if { has_avx512f() } else { false }),
            Self::Neon => cfg_neon!(if { true } else { false }),
        }
    }
}

/// The backend `dispatch!` selects: the fastest available one.
pub(crate) fn backend() -> Backend {
    dispatch! {
        avx512 => Backend::Avx512,
        avx2 => Backend::Avx2,
        neon => Backend::Neon,
        scalar => Backend::Scalar,
    }
}

cfg_x86! {
    /// AVX2 and FMA, required by the AVX2 backend.
    #[inline(always)]
    pub(crate) fn has_avx2_fma() -> bool {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(all(target_feature = "avx2", target_feature = "fma"))
        }
    }

    /// AVX-512F, required by the AVX-512 backend.
    #[inline(always)]
    pub(crate) fn has_avx512f() -> bool {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = "avx512f")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_backend_is_available() {
        assert!(backend().is_available());
    }

    cfg_x86! {
        #[target_feature(enable = "avx512f")]
        fn on_avx512() -> Backend {
            Backend::Avx512
        }

        #[target_feature(enable = "avx2,fma")]
        fn on_avx2() -> Backend {
            Backend::Avx2
        }
    }

    cfg_neon! {
        fn on_neon() -> Backend {
            Backend::Neon
        }
    }

    #[test]
    fn dispatch_calls_the_selected_backend() {
        let got = dispatch! {
            avx512 => on_avx512(),
            avx2 => on_avx2(),
            neon => on_neon(),
            scalar => Backend::Scalar,
        };
        assert_eq!(got, backend());
    }

    #[test]
    fn dispatch_skips_missing_arms() {
        let got = dispatch! {
            avx2 => on_avx2(),
            scalar => Backend::Scalar,
        };
        let want = if Backend::Avx2.is_available() {
            Backend::Avx2
        } else {
            Backend::Scalar
        };
        assert_eq!(got, want);
    }

    /// CI lists the backends each job must exercise, e.g. `avx2,avx512`, so
    /// that backend tests cannot pass by silently skipping.
    #[test]
    fn required_backends_are_available() {
        let Ok(required) = std::env::var("SIMDLY_REQUIRE_BACKENDS") else {
            return;
        };
        for name in required.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let backend = match name {
                "scalar" => Backend::Scalar,
                "avx2" => Backend::Avx2,
                "avx512" => Backend::Avx512,
                "neon" => Backend::Neon,
                _ => panic!("unknown backend {name:?} in SIMDLY_REQUIRE_BACKENDS"),
            };
            assert!(
                backend.is_available(),
                "required backend {name} is not available"
            );
        }
    }
}
