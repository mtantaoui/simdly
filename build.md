# CPU Feature Detection and Optimization

This project includes an advanced build-time CPU feature detection system that automatically enables optimal SIMD instructions for your hardware. The `build.rs` script detects available CPU features and configures the compiler to use the best available instruction sets.

## 🚀 Features

- **Automatic CPU Detection**: Detects SSE4.1, AVX2, AVX512F, and NEON instruction sets
- **Cross-Platform Support**: Works on Linux, macOS, and Windows
- **Smart Fallbacks**: Graceful degradation when detection fails
- **Cross-Compilation Safe**: Automatically uses fallback for cross-compilation
- **Nightly Support**: Enables AVX512 optimizations on nightly Rust builds

## 🏗️ How It Works

### Detection Process

The build system follows this detection hierarchy (highest to lowest priority):

1. **AVX512F** (Nightly only) - Latest Intel server/HEDT processors
2. **AVX2** - Modern Intel/AMD processors (2013+)
3. **SSE4.1** - Most x86_64 processors (2007+)
4. **NEON** - ARM processors (mobile, Apple Silicon)
5. **Fallback** - Pure Rust implementation for all other cases

### Platform-Specific Detection

#### Linux

```bash
# Reads /proc/cpuinfo for feature flags
cat /proc/cpuinfo | grep flags
```

#### macOS

```bash
# Uses sysctl to query hardware capabilities
sysctl -a | grep hw.optional
```

#### Windows

- **Primary**: PowerShell with WMI queries for detailed CPU information
- **Fallback**: Traditional `wmic` command with heuristic detection
- **Smart Detection**: Uses processor names and generations to determine capabilities

### Build Configuration

The system automatically configures these compiler flags based on detection:

```toml
# Example configurations applied
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+avx,+fma"]  # AVX2 detected

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-feature=+neon"]             # Apple Silicon
```

## 🔧 Build Output

During compilation, you'll see informative messages:

```bash
warning: Using Linux CPU feature detector
warning: CPU feature detection successful
warning: Enabled CPU feature optimization: avx2 (+avx2,+avx,+fma)
```

Or for fallback scenarios:

```bash
warning: Cross-compiling detected (host: x86_64-unknown-linux-gnu, target: aarch64-unknown-linux-gnu)
warning: Skipping CPU feature detection, using fallback implementation
warning: No CPU features detected, using fallback implementation
```

## 🛠️ Advanced Configuration

### Environment Variables

- `RUSTC`: Override the Rust compiler path
- `HOST`: Build host architecture (auto-detected)
- `TARGET`: Target architecture (auto-detected)

### Nightly Features

To enable AVX512 support, use nightly Rust:

```bash
rustup install nightly
cargo +nightly build
```

### Cross-Compilation

The system automatically detects cross-compilation and uses safe defaults:

```bash
# Cross-compiling - will use fallback
cargo build --target aarch64-unknown-linux-gnu

# Native compilation - will detect features
cargo build
```

## 🔒 Safety and Reliability

### Fallback Guarantee

The system is designed to **never fail your build**. If CPU detection fails for any reason, it gracefully falls back to a pure Rust implementation that works on all platforms.

### Cross-Compilation Safety

When cross-compiling, CPU detection is automatically disabled to prevent issues with detecting the wrong architecture's features.

### Error Handling

All system calls are wrapped in proper error handling with detailed logging to help diagnose issues without breaking the build process.

## 📚 Technical Details

### Architecture

The detection system uses a modular architecture:

```
PlatformDetector
├── LinuxDetector      (reads /proc/cpuinfo)
├── MacOSDetector      (uses sysctl)
└── WindowsDetector    (PowerShell + wmic fallback)
```

### Priority System

Features are prioritized to select the most advanced available:

1. AVX512F (priority 0) - highest
2. AVX2 (priority 1)
3. SSE4.1 (priority 2)
4. Others (priority MAX) - lowest

### Detection Results

The system returns structured results:

- `Success`: All features detected correctly
- `PartialFailure`: Some detection worked with warnings
- `Failure`: Complete failure, uses fallback

This ensures you always get feedback about what happened during detection while maintaining build reliability.
