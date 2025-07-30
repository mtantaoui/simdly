---
title: Installation
description: How to install and set up Simdly in your cross-platform Rust project.
---

# Installation

## Prerequisites

- **Rust**: Version 1.77 or later
- **Target Architecture**: x86/x86_64 (AVX2) or ARM/AArch64 (NEON)
- **Operating System**: Linux, macOS, or Windows

## Adding Simdly to Your Project

Add Simdly to your `Cargo.toml`:

```toml
[dependencies]
simdly = "0.1.2"
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

### Platform-Specific Overrides (Advanced)

For specific deployment scenarios:

```bash
# Override for specific targets
export RUSTFLAGS="-C target-feature=+avx2"        # Force AVX2 on x86
export RUSTFLAGS="-C target-feature=+neon"        # Force NEON on ARM
cargo build --release
```

### Cross-Compilation

For cross-compilation to different architectures:

```bash
# Cross-compile to ARM64 from x86
cargo build --release --target aarch64-unknown-linux-gnu

# Cross-compile to x86_64 from ARM
cargo build --release --target x86_64-unknown-linux-gnu
```

## Verification

To verify that Simdly is working correctly, create a simple test:

```rust
use simdly::f32x8;

fn main() {
    // Test basic vector operations
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = f32x8::load_unaligned(&data);
    let doubled = vec.mul(f32x8::splat(2.0));
    
    println!("Simdly is ready! Vector operations working on {} lanes", f32x8::lanes());
}
```

## Performance Considerations

- **Automatic Optimization**: Simdly's `.cargo/config.toml` automatically enables the best SIMD features for your CPU
- **Release Mode**: Always use `--release` for performance testing
- **Memory Alignment**: Use 32-byte alignment on x86, 16-byte on ARM for optimal performance
- **Cross-Compilation**: Simdly supports cross-compilation between x86 and ARM targets
- **No Setup Required**: Just `cargo add simdly` and start coding - SIMD optimization is automatic!

## Next Steps

Now that you have Simdly installed, check out the [Quick Start Guide](/getting-started/quick-start/) to learn basic cross-platform usage patterns.