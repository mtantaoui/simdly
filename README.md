# Simdly

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
simdly = "0.1.7"
```

For optimal performance, enable AVX2 support:

```toml
[build]
rustflags = ["-C", "target-feature=+avx2"]
```

## 🚀 Quick Start

### Simple Vector Addition with Multiple Algorithms

```rust
use simdly::SimdAdd;

fn main() {
    // Create two vectors
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    
    // Choose the appropriate algorithm based on your needs:
    
    // For small arrays (< 128 elements)
    let result = a.as_slice().scalar_add(b.as_slice());
    
    // For medium arrays (128+ elements) - uses SIMD
    let result = a.as_slice().simd_add(b.as_slice());
    
    // For large arrays (262,144+ elements) - uses parallel SIMD
    let result = a.as_slice().par_simd_add(b.as_slice());
    
    println!("Result: {:?}", result); // [3.0, 5.0, 7.0, 9.0, 11.0]
}
```

### Working with SIMD Vectors Directly

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;
use simdly::simd::{SimdLoad, SimdStore};

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        // Load 8 f32 values into AVX2 SIMD vector
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vec = F32x8::from(&data[..]);
        
        // Store results using platform-appropriate method
        let mut output = [0.0f32; 8];
        unsafe {
            vec.store_at(output.as_mut_ptr());
        }
        
        println!("Processed {} elements with AVX2 SIMD", vec.size);
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // Load 4 f32 values into NEON SIMD vector
        let data = [1.0, 2.0, 3.0, 4.0];
        let vec = F32x4::from(&data[..]);
        
        // Store results
        let mut output = [0.0f32; 4];
        unsafe {
            vec.store_at(output.as_mut_ptr());
        }
        
        println!("Processed {} elements with NEON SIMD", vec.size);
    }
}
```

### Working with Partial Data

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;
use simdly::simd::{SimdLoad, SimdStore};

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        // Handle arrays smaller than 8 elements
        let data = [1.0, 2.0, 3.0]; // Only 3 elements
        let vec = F32x8::from(&data[..]);

        let mut output = [0.0f32; 8];
        unsafe {
            vec.store_at_partial(output.as_mut_ptr());
        }
        // Only first 3 elements are written
        println!("Partial AVX2: {:?}", &output[..3]);
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // Handle arrays smaller than 4 elements
        let data = [1.0, 2.0]; // Only 2 elements
        let vec = F32x4::from(&data[..]);

        let mut output = [0.0f32; 4];
        unsafe {
            vec.store_at_partial(output.as_mut_ptr());
        }
        // Only first 2 elements are written
        println!("Partial NEON: {:?}", &output[..2]);
    }
}
```

### Mathematical Operations

```rust
#[cfg(target_arch = "x86_64")]
{
    use simdly::simd::avx2::math::{_mm256_sin_ps, _mm256_hypot_ps};
    use std::arch::x86_64::_mm256_set1_ps;

    // 8 parallel sine calculations
    let input = _mm256_set1_ps(1.0);
    let result = unsafe { _mm256_sin_ps(input) };

    // 2D Euclidean distance for 8 point pairs
    let x = _mm256_set1_ps(3.0);
    let y = _mm256_set1_ps(4.0);
    let distance = unsafe { _mm256_hypot_ps(x, y) }; // sqrt(3² + 4²) = 5.0
}
```

### High-Level Mathematical Operations

```rust
use simdly::simd::SimdMath;

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    
    // All mathematical operations use SIMD automatically
    let cosines = data.cos();              // Vectorized cosine
    let sines = data.sin();                // Vectorized sine
    let exponentials = data.exp();         // Vectorized exponential
    let square_roots = data.sqrt();        // Vectorized square root
    
    // Power and distance operations
    let base = vec![2.0, 3.0, 4.0, 5.0];
    let exp = vec![2.0, 2.0, 2.0, 2.0];
    let powers = base.pow(exp);            // Powers: [4.0, 9.0, 16.0, 25.0]
    
    let x = vec![3.0, 5.0, 8.0, 7.0];
    let y = vec![4.0, 12.0, 15.0, 24.0];
    let distances = x.hypot(y);            // 2D distances: [5.0, 13.0, 17.0, 25.0]
    
    println!("Results computed with SIMD acceleration!");
}
```

## 📊 Performance

simdly provides significant performance improvements for numerical computations with multiple algorithm options:

### Algorithm Selection

The `SimdAdd` trait provides multiple algorithms that you can choose based on your data size:

| Array Size Range | Recommended Method | Algorithm | Rationale |
|------------------|-------------------|-----------|-----------|
| < 128 elements | `scalar_add()` | **Scalar** | Avoids SIMD setup overhead |
| 128 - 262,143 elements | `simd_add()` | **SIMD** | Optimal vectorization benefits |
| ≥ 262,144 elements | `par_simd_add()` | **Parallel SIMD** | Memory bandwidth + multi-core scaling |

### Performance Characteristics

- **Mathematical Operations**: SIMD shows 4x-13x speedup for complex operations like cosine
- **Simple Operations**: Intelligent thresholds prevent performance regression on small arrays
- **Memory Hierarchy**: Optimized chunk sizes (16 KiB) for L1 cache efficiency
- **Cross-Platform**: Thresholds work optimally on Intel AVX2 and ARM NEON architectures

### Benchmark Results (Addition)

Performance measurements on modern x64 with AVX2:

| Vector Size | Elements | Recommended Method | Performance Benefit |
|-------------|----------|-------------------|---------------------|
| 512 B | 128 | `scalar_add()` | Baseline (no overhead) |
| 20 KiB | 5,000 | `simd_add()` | ~4-8x throughput |
| 1 MiB | 262,144 | `par_simd_add()` | ~4-8x × cores |
| 4 MiB | 1,048,576 | `par_simd_add()` | Memory bandwidth limited |

### Mathematical Functions Performance

Complex mathematical operations benefit from SIMD across all sizes:

| Function | Array Size | SIMD Speedup | Notes |
|----------|------------|--------------|-------|
| `cos()` | 4 KiB | 4.4x | Immediate benefit |
| `cos()` | 64 KiB | 11.7x | Peak efficiency |
| `cos()` | 1 MiB | 13.3x | Best performance |
| `cos()` | 128 MiB | 9.2x | Memory-bound |

### Key Features

- **Manual Optimization**: Choose the best algorithm for your specific use case
- **Zero-Cost Abstraction**: Direct method calls with no runtime overhead
- **Memory Efficiency**: Cache-aware chunking and aligned memory access
- **Scalable Performance**: Near-linear scaling with available CPU cores

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

### Manual Algorithm Selection with SimdAdd

simdly provides multiple algorithms that you can choose based on your specific needs:

```rust
use simdly::SimdAdd;

fn main() {
    // Small arrays (< 128 elements) - use scalar addition
    let small_a = vec![1.0; 100];
    let small_b = vec![2.0; 100];
    let result = small_a.as_slice().scalar_add(small_b.as_slice());
    
    // Medium arrays (128 - 262,143 elements) - use SIMD
    let medium_a = vec![1.0; 5_000];
    let medium_b = vec![2.0; 5_000];
    let result = medium_a.as_slice().simd_add(medium_b.as_slice());
    
    // Large arrays (≥ 262,144 elements) - use parallel SIMD
    let large_a = vec![1.0; 300_000];
    let large_b = vec![2.0; 300_000];
    let result = large_a.as_slice().par_simd_add(large_b.as_slice());
}
```

### Manual Algorithm Selection

For fine-grained control, you can manually select the algorithm:

```rust
use simdly::SimdAdd;

fn main() {
    let a = vec![1.0; 10_000];
    let b = vec![2.0; 10_000];
    
    // Force scalar addition
    let scalar_result = a.as_slice().scalar_add(b.as_slice());
    
    // Force SIMD addition
    let simd_result = a.as_slice().simd_add(b.as_slice());
    
    // Force parallel SIMD addition
    let parallel_result = a.as_slice().par_simd_add(b.as_slice());
}
```

### Mathematical Operations with SIMD

```rust
use simdly::simd::SimdMath;

fn main() {
    // Vectorized cosine computation
    let angles = vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0];
    let cosines = angles.as_slice().cos(); // Uses SIMD automatically
    
    println!("cos(0) = {}", cosines[0]);        // ≈ 1.0
    println!("cos(π/4) = {}", cosines[1]);      // ≈ 0.707
    println!("cos(π/2) = {}", cosines[2]);      // ≈ 0.0
}
```

### Processing Large Arrays

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;
use simdly::simd::{SimdLoad, SimdStore, SimdMath};

fn process_array(input: &[f32]) -> Vec<f32> {
    // For real applications, use high-level SIMD operations
    input.cos() // Vectorized cosine computation
}

#[cfg(target_arch = "x86_64")]
fn manual_avx2_processing(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    // Process full chunks of 8 elements
    for (i, chunk) in input.chunks_exact(8).enumerate() {
        let vec = F32x8::from(chunk);
        
        // Example: compute cosine using SIMD
        let result = vec.cos();
        
        unsafe {
            result.store_at(output[i * 8..].as_mut_ptr());
        }
    }
    
    // Handle remaining elements
    let remainder_start = (input.len() / 8) * 8;
    if remainder_start < input.len() {
        let vec = F32x8::from(&input[remainder_start..]);
        let result = vec.cos();
        
        unsafe {
            result.store_at_partial(output[remainder_start..].as_mut_ptr());
        }
    }
    
    output
}

#[cfg(target_arch = "aarch64")]
fn manual_neon_processing(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    // Process full chunks of 4 elements  
    for (i, chunk) in input.chunks_exact(4).enumerate() {
        let vec = F32x4::from(chunk);
        
        // Example: compute cosine using SIMD
        let result = vec.cos();
        
        unsafe {
            result.store_at(output[i * 4..].as_mut_ptr());
        }
    }
    
    // Handle remaining elements
    let remainder_start = (input.len() / 4) * 4;
    if remainder_start < input.len() {
        let vec = F32x4::from(&input[remainder_start..]);
        let result = vec.cos();
        
        unsafe {
            result.store_at_partial(output[remainder_start..].as_mut_ptr());
        }
    }
    
    output
}
```

### Memory-Aligned Operations

```rust
#[cfg(target_arch = "x86_64")]
use simdly::simd::avx2::f32x8::F32x8;
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::f32x4::F32x4;
use simdly::simd::{Alignment, SimdLoad, SimdStore};
use std::alloc::{alloc, dealloc, Layout};

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        // Allocate 32-byte aligned memory for AVX2
        let layout = Layout::from_size_align(8 * std::mem::size_of::<f32>(), 32).unwrap();
        let aligned_ptr = unsafe { alloc(layout) as *mut f32 };

        // Verify alignment
        assert!(F32x8::is_aligned(aligned_ptr));

        // Use standard load/store (AVX2 handles alignment automatically)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), aligned_ptr, 8);
            
            let vec = F32x8::from(std::slice::from_raw_parts(aligned_ptr, 8));
            vec.store_at(aligned_ptr);
        }

        // Clean up
        unsafe { dealloc(aligned_ptr as *mut u8, layout) };
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // NEON doesn't require special alignment handling
        let data = [1.0, 2.0, 3.0, 4.0];
        let vec = F32x4::from(&data[..]);
        
        let mut output = [0.0f32; 4];
        unsafe {
            vec.store_at(output.as_mut_ptr());
        }
        
        println!("NEON handles alignment automatically");
    }
}
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

### Performance Benchmarks

The crate includes comprehensive benchmarks showing real-world performance improvements:

```bash
# Run benchmarks to measure performance on your hardware
cargo bench

# View detailed benchmark reports
open target/criterion/report/index.html
```

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
