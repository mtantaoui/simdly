# Simdly

> ⚠️ **Development Status**: This project is currently under active development. APIs may change and features are still being implemented.

🚀 A high-performance Rust library that leverages SIMD (Single Instruction, Multiple Data) instructions for fast vectorized computations. This library provides efficient implementations of mathematical operations using modern CPU features.

[![Crates.io](https://img.shields.io/crates/v/simdly.svg)](https://crates.io/crates/simdly)
[![Documentation](https://docs.rs/simdly/badge.svg)](https://docs.rs/simdly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-blue.svg)](https://www.rust-lang.org/)

## ✨ Features

- **🚀 SIMD Optimized**: Leverages AVX2 (256-bit) and NEON (128-bit) instructions for vector operations
- **🧠 Intelligent Algorithm Selection**: Automatic choice between scalar, SIMD, and parallel algorithms based on data size
- **💾 Memory Efficient**: Supports both aligned and unaligned memory access patterns with cache-aware chunking
- **🔧 Generic Traits**: Provides consistent interfaces across different SIMD implementations
- **🛡️ Safe Abstractions**: Wraps unsafe SIMD operations in safe, ergonomic APIs with robust error handling
- **🧮 Rich Math Library**: Extensive mathematical functions (trig, exp, log, sqrt, etc.) with SIMD acceleration
- **⚡ Performance**: Optimized thresholds prevent overhead while maximizing throughput gains

## 🏗️ Architecture Support

### Currently Supported

- **x86/x86_64** with AVX2 (256-bit vectors)
- **ARM/AArch64** with NEON (128-bit vectors)

### Planned Support

- SSE (128-bit vectors for older x86 processors)

## 📦 Installation

Add simdly to your `Cargo.toml`:

```toml
[dependencies]
simdly = "0.1.10"
```

For optimal performance, enable AVX2 support in your build configuration.

## 🚀 Quick Start

The library provides multiple algorithms for vector operations that you can choose based on your data size:

- **Small arrays (< 128 elements)**: Use scalar operations to avoid SIMD setup overhead
- **Medium arrays (128+ elements)**: Use SIMD operations for optimal vectorization benefits
- **Large arrays (≥ 262,144 elements)**: Use parallel SIMD for memory bandwidth and multi-core scaling

The library supports working with SIMD vectors directly, handling partial data efficiently, and provides mathematical operations with automatic SIMD acceleration including trigonometric functions, exponentials, square roots, powers, and distance calculations.

## 📊 Performance

simdly provides significant performance improvements for numerical computations with multiple algorithm options:

### Algorithm Selection

### Performance Characteristics

- **Mathematical Operations**: SIMD shows 4x-13x speedup for complex operations like cosine
- **Simple Operations**: Intelligent thresholds prevent performance regression on small arrays
- **Memory Hierarchy**: Optimized chunk sizes (16 KiB) for L1 cache efficiency
- **Cross-Platform**: Thresholds work optimally on Intel AVX2 and ARM NEON architectures

### Mathematical Functions Performance

Complex mathematical operations benefit from SIMD across all sizes:

| Function | Array Size | SIMD Speedup | Notes             |
| -------- | ---------- | ------------ | ----------------- |
| `cos()`  | 4 KiB      | 4.4x         | Immediate benefit |
| `cos()`  | 64 KiB     | 11.7x        | Peak efficiency   |
| `cos()`  | 1 MiB      | 13.3x        | Best performance  |
| `cos()`  | 128 MiB    | 9.2x         | Memory-bound      |

### Key Features

- **Manual Optimization**: Choose the best algorithm for your specific use case
- **Zero-Cost Abstraction**: Direct method calls with no runtime overhead
- **Memory Efficiency**: Cache-aware chunking and aligned memory access
- **Scalable Performance**: Near-linear scaling with available CPU cores

### Compilation Flags

For maximum performance, compile with target-feature flags for AVX2 support, and consider using link-time optimization (LTO) and single codegen unit configuration in your release profile.

## 🔧 Usage

The library provides multiple algorithms that you can choose based on your specific needs, with fine-grained control over algorithm selection. It supports vectorized mathematical operations with automatic SIMD acceleration, efficient processing of large arrays with chunking strategies, and memory-aligned operations for optimal performance on both AVX2 and NEON architectures.

## 📚 Documentation

- **📖 [API Documentation](https://docs.rs/simdly/)** - Complete API reference
- **🚀 [Getting Started Guide](docs/)** - Detailed usage examples and tutorials
- **⚡ [Performance Tips](docs/)** - Optimization strategies and best practices

## 🛠️ Development

### Prerequisites

- Rust 1.77 or later
- x86/x86_64 processor with AVX2 support
- Linux, macOS, or Windows

### Building

Clone the repository and build with `cargo build --release`.

### Testing

Run tests with `cargo test`.

### Performance Benchmarks

The crate includes comprehensive benchmarks showing real-world performance improvements:

Run benchmarks with `cargo bench` and view detailed reports in the target/criterion/report/ directory.

**Key Findings from Benchmarks:**

- Mathematical operations (`cos`, `sin`, `exp`, etc.) show significant SIMD acceleration
- Parallel methods automatically optimize based on array size using `PARALLEL_SIMD_THRESHOLD`
- Performance varies by CPU architecture - benchmarks show actual improvements on your hardware

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution

- Additional SIMD instruction set support (SSE)
- Advanced mathematical operations implementation
- Performance optimizations and micro-benchmarks
- Documentation improvements and examples
- Testing coverage and edge case validation
- WebAssembly SIMD support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Rust's excellent SIMD intrinsics
- Inspired by high-performance computing libraries
- Thanks to the Rust community for their valuable feedback

## 📈 Roadmap

- [x] **ARM NEON support for ARM/AArch64** - ✅ Complete with full mathematical operations
- [x] **Additional mathematical operations** - ✅ Power, 2D/3D/4D hypotenuse, and more
- [ ] SSE support for older x86 processors
- [ ] Automatic SIMD instruction set detection
- [ ] WebAssembly SIMD support
- [ ] Additional mathematical functions (bessel, gamma, etc.)
- [ ] Complex number SIMD operations

---

**Made with ❤️ and ⚡ by [Mahdi Tantaoui](https://github.com/mtantaoui)**
