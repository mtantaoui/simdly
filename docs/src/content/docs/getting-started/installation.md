---
title: Installation
description: How to install and set up Simdly in your cross-platform Rust project.
sidebar:
  order: 2
---

## Prerequisites

- **Rust**: Version 1.77 or later
- **Target Architecture**: x86/x86_64 (AVX2) or ARM/AArch64 (NEON)
- **Operating System**: Linux, macOS, or Windows

## Adding Simdly to Your Project

Add Simdly to your `Cargo.toml`:

```toml
[dependencies]
simdly = "0.1.10"
```

## Automatic SIMD Optimization

Simdly comes pre-configured with optimal SIMD settings! The crate includes a `.cargo/config.toml` that automatically enables native CPU features:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

This means **no additional configuration is needed** - Simdly will automatically use the best SIMD instructions available on your CPU (AVX2 on x86/x86_64, NEON on ARM).

## Custom Configuration (Optional)

If you need to override the default settings or use Simdly in a different project, here are the available options:

### For Your Own Projects

If you want to use similar settings in your own projects:

```toml
# In your project's .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

### Platform-Specific Override

For specific deployment scenarios:

```bash
# Override for specific targets
export RUSTFLAGS="-C target-feature=+avx2"        # Force AVX2 on x86
export RUSTFLAGS="-C target-feature=+neon"        # Force NEON on ARM
cargo build --release
```

## Performance Considerations

- **Automatic Optimization**: Simdly automatically detects CPU features and uses the best SIMD instructions available
- **Release Mode**: Always use `cargo build --release` for performance testing and production
- **Data Size Thresholds**:
  - **Small arrays** (<128 elements): Uses scalar operations for lower overhead
  - **Medium arrays** (128+ elements): Uses SIMD operations for speed
  - **Large arrays** (262,144+ elements): Uses parallel SIMD across multiple cores
- **Cross-Platform Support**: Same API works on both x86_64 (AVX2) and ARM64 (NEON)
- **Zero Configuration**: Just `cargo add simdly` and start coding - optimization is automatic!

## Next Steps

Now that you have Simdly installed, check out the [Quick Start Guide](/getting-started/quick-start/) to learn basic cross-platform usage patterns.
