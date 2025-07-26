---
title: Installation
description: How to install and set up simdly in your Rust project.
---

# Installation

## Prerequisites

- **Rust**: Version 1.77 or later
- **Target Architecture**: x86/x86_64 with AVX2 support
- **Operating System**: Linux, macOS, or Windows

## Adding simdly to Your Project

Add simdly to your `Cargo.toml`:

```toml
[dependencies]
simdly = "0.1.2"
```

## Enabling Target Features

For optimal performance, enable AVX2 target features. You can do this in several ways:

### Option 1: Cargo.toml (Recommended)

Add to your `Cargo.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+avx2"]
```

### Option 2: Environment Variable

Set the `RUSTFLAGS` environment variable:

```bash
export RUSTFLAGS="-C target-feature=+avx2"
cargo build --release
```

### Option 3: Command Line

Build with target features directly:

```bash
cargo build --release -C target-feature=+avx2
```

## Verification

To verify that simdly is working correctly, create a simple test:

```rust
use simdly::simd::SimdVec;

fn main() {
    // Your simdly code here
    println!("simdly is ready to use!");
}
```

## Performance Considerations

- **Release Mode**: Always use `--release` for performance testing
- **Target Features**: Enable AVX2 for best performance
- **Memory Alignment**: Consider using aligned allocators for optimal performance

## Next Steps

Now that you have simdly installed, check out the [Quick Start Guide](/getting-started/quick-start/) to learn basic usage patterns.