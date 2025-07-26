# simdly

🚀 A high-performance Rust library that leverages SIMD (Single Instruction, Multiple Data) instructions for fast vectorized computations. This library provides efficient implementations of mathematical operations using modern CPU features.

[![Crates.io](https://img.shields.io/crates/v/simdly.svg)](https://crates.io/crates/simdly)
[![Documentation](https://docs.rs/simdly/badge.svg)](https://docs.rs/simdly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-blue.svg)](https://www.rust-lang.org/)

## ✨ Features

- **🚀 SIMD Optimized**: Leverages AVX2 instructions for 256-bit vector operations
- **💾 Memory Efficient**: Supports both aligned and unaligned memory access patterns
- **🔧 Generic Traits**: Provides consistent interfaces across different SIMD implementations
- **🛡️ Safe Abstractions**: Wraps unsafe SIMD operations in safe, ergonomic APIs
- **⚡ Performance**: Optimized for high-throughput numerical computations

## 🏗️ Architecture Support

### Currently Supported

- **x86/x86_64** with AVX2 (256-bit vectors)

### Planned Support

- SSE (128-bit vectors for older x86 processors)
- ARM NEON (128-bit vectors for ARM/AArch64)

## 📦 Installation

Add simdly to your `Cargo.toml`:

```toml
[dependencies]
simdly = "0.1.3"
```

For optimal performance, enable AVX2 support:

```toml
[build]
rustflags = ["-C", "target-feature=+avx2"]
```

## 🚀 Quick Start

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

fn main() {
    // Load 8 f32 values into SIMD vector
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = F32x8::from_slice(&data);
    
    // Store results
    let mut output = [0.0f32; 8];
    unsafe {
        vec.store_unaligned_at(output.as_mut_ptr());
    }
    
    println!("Processed {} elements with SIMD", vec.size);
}
```

### Working with Partial Data

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

// Handle arrays smaller than 8 elements
let data = [1.0, 2.0, 3.0]; // Only 3 elements
let vec = F32x8::from_slice(&data);

let mut output = [0.0f32; 8];
unsafe {
    vec.store_at_partial(output.as_mut_ptr());
}
// Only first 3 elements are written
```

## 📊 Performance

simdly can provide significant performance improvements for numerical computations:

- **Up to 8x faster** operations using AVX2 256-bit vectors
- **Memory bandwidth optimization** through aligned memory access
- **Cache-friendly** processing patterns

### Compilation Flags

For maximum performance, compile with:

```bash
RUSTFLAGS="-C target-feature=+avx2" cargo build --release
```

Or add to your `Cargo.toml`:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
```

## 🔧 Usage Examples

### Processing Large Arrays

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{SimdLoad, SimdStore};

fn process_array(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    // Process full chunks of 8 elements
    for (i, chunk) in input.chunks_exact(8).enumerate() {
        let vec = F32x8::from_slice(chunk);
        
        // Your SIMD operations here...
        
        unsafe {
            vec.store_unaligned_at(output[i * 8..].as_mut_ptr());
        }
    }
    
    // Handle remaining elements
    let remainder_start = (input.len() / 8) * 8;
    if remainder_start < input.len() {
        let vec = F32x8::from_slice(&input[remainder_start..]);
        
        unsafe {
            vec.store_at_partial(output[remainder_start..].as_mut_ptr());
        }
    }
    
    output
}
```

### Memory-Aligned Operations

```rust
use simdly::simd::avx2::f32x8::F32x8;
use simdly::simd::{Alignment, SimdLoad, SimdStore};
use std::alloc::{alloc, dealloc, Layout};

// Allocate 32-byte aligned memory for optimal performance
let layout = Layout::from_size_align(8 * std::mem::size_of::<f32>(), 32).unwrap();
let aligned_ptr = unsafe { alloc(layout) as *mut f32 };

// Verify alignment
assert!(F32x8::is_aligned(aligned_ptr));

// Use aligned operations for best performance
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
unsafe {
    std::ptr::copy_nonoverlapping(data.as_ptr(), aligned_ptr, 8);
    
    let vec = F32x8::load_aligned(aligned_ptr);
    vec.store_aligned_at(aligned_ptr);
}

// Clean up
unsafe { dealloc(aligned_ptr as *mut u8, layout) };
```

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

```bash
git clone https://github.com/mtantaoui/simdly.git
cd simdly
cargo build --release
```

### Testing

```bash
cargo test
```

### Benchmarking

```bash
cargo bench
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution

- Additional SIMD instruction set support (SSE, ARM NEON)
- Mathematical operations implementation
- Performance optimizations
- Documentation improvements
- Testing and benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Rust's excellent SIMD intrinsics
- Inspired by high-performance computing libraries
- Thanks to the Rust community for their valuable feedback

## 📈 Roadmap

- [ ] SSE support for older x86 processors
- [ ] ARM NEON support for ARM/AArch64
- [ ] Additional mathematical operations
- [ ] Automatic SIMD instruction set detection
- [ ] WebAssembly SIMD support

---

**Made with ❤️ and ⚡ by [Mahdi Tantaoui](https://github.com/mtantaoui)**
