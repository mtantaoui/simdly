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
simdly = "0.1.6"
```

For optimal performance, enable AVX2 support:

```toml
[build]
rustflags = ["-C", "target-feature=+avx2"]
```

## 🚀 Quick Start

### Simple Vector Addition with Automatic Optimization

```rust
use simdly::FastAdd;

fn main() {
    // Create two vectors
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    
    // FastAdd automatically chooses the best algorithm
    let result = a.as_slice().fast_add(b.as_slice());
    
    println!("Result: {:?}", result); // [3.0, 5.0, 7.0, 9.0, 11.0]
}
```

### Working with SIMD Vectors Directly

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

### Error Handling

simdly uses robust error handling instead of panics:

```rust
use simdly::{SimdAdd, error::SimdlyError};

fn safe_computation(a: &[f32], b: &[f32]) -> Result<Vec<f32>, SimdlyError> {
    // All operations return Results for graceful error handling
    a.simd_add(b)
}

// Handle different error types
match a.simd_add(b) {
    Ok(result) => println!("Success: {:?}", result),
    Err(SimdlyError::ValidationError { message }) => {
        eprintln!("Input validation failed: {}", message);
    }
    Err(SimdlyError::AllocationError { message, .. }) => {
        eprintln!("Memory allocation failed: {}", message);
    }
    Err(e) => eprintln!("Operation failed: {}", e),
}
```

## 📊 Performance

simdly provides significant performance improvements for numerical computations with intelligent algorithm selection:

### Intelligent Algorithm Selection

The `FastAdd` trait automatically selects the optimal algorithm based on empirically determined thresholds:

| Array Size Range | Algorithm | Rationale |
|------------------|-----------|-----------|
| < 256 elements | **Scalar** | Avoids SIMD setup overhead |
| 256 - 131,071 elements | **SIMD** | Optimal vectorization benefits |
| ≥ 131,072 elements | **Parallel SIMD** | Memory bandwidth + multi-core scaling |

### Performance Characteristics

- **Mathematical Operations**: SIMD shows 4x-13x speedup for complex operations like cosine
- **Simple Operations**: Intelligent thresholds prevent performance regression on small arrays
- **Memory Hierarchy**: Optimized chunk sizes (16 KiB) for L1 cache efficiency
- **Cross-Platform**: Thresholds work optimally on Intel AVX2 and ARM NEON architectures

### Benchmark Results (Addition)

Performance measurements on modern x64 with AVX2:

| Vector Size | Elements | FastAdd Strategy | Performance Benefit |
|-------------|----------|------------------|---------------------|
| 1 KiB | 256 | Scalar | Baseline (no overhead) |
| 20 KiB | 5,000 | SIMD | ~4-8x throughput |
| 512 KiB | 131,072 | Parallel SIMD | ~4-8x × cores |
| 4 MiB | 1,048,576 | Parallel SIMD | Memory bandwidth limited |

### Mathematical Functions Performance

Complex mathematical operations benefit from SIMD across all sizes:

| Function | Array Size | SIMD Speedup | Notes |
|----------|------------|--------------|-------|
| `cos()` | 4 KiB | 4.4x | Immediate benefit |
| `cos()` | 64 KiB | 11.7x | Peak efficiency |
| `cos()` | 1 MiB | 13.3x | Best performance |
| `cos()` | 128 MiB | 9.2x | Memory-bound |

### Key Features

- **Automatic Optimization**: `FastAdd` chooses the best algorithm without manual tuning
- **Zero-Cost Abstraction**: Intelligent selection with no runtime overhead
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

### Intelligent Algorithm Selection with FastAdd

simdly provides intelligent algorithm selection that automatically chooses the optimal addition strategy based on input size:

```rust
use simdly::FastAdd;

fn main() {
    // Small arrays (< 256 elements) - uses scalar addition
    let small_a = vec![1.0; 100];
    let small_b = vec![2.0; 100];
    let result = small_a.as_slice().fast_add(small_b.as_slice());
    
    // Medium arrays (256 - 131,071 elements) - uses SIMD
    let medium_a = vec![1.0; 5_000];
    let medium_b = vec![2.0; 5_000];
    let result = medium_a.as_slice().fast_add(medium_b.as_slice());
    
    // Large arrays (≥ 131,072 elements) - uses parallel SIMD
    let large_a = vec![1.0; 200_000];
    let large_b = vec![2.0; 200_000];
    let result = large_a.as_slice().fast_add(large_b.as_slice());
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
