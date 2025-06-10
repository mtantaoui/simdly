# Simdly

Simdly is a Rust project focused on exploring and utilizing SIMD (Single Instruction, Multiple Data) operations for high-performance computing. Leveraging modern Rust features and libraries like Rayon and Criterion for benching, Simdly aims to provide or demonstrate efficient numerical computations.

## Table of Contents

- [Project Status](#project-status)
- [Platform Support](#platform-support)
- [SIMD, Alignment, and Rust's `Vec<T>`/Slices](#simd-alignment-and-rusts-vectslices)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Cloning the Repository](#cloning-the-repository)
  - [Building the Project](#building-the-project)
- [Usage](#usage)
- [Building](#building)
  - [Development Build](#development-build)
  - [Release Build](#release-build)
- [Benchmarking Environment](#benchmarking-environment)
- [Running Benchmarks](#running-benchmarks)
  - [Overview](#overview)
  - [Available Benchmarks](#available-benchmarks)
  - [How to Run](#how-to-run)
  - [Benchmark Reports and Performance](#benchmark-reports-and-performance)
    - [Summary of `add` Benchmark Results](#summary-of-add-benchmark-results)
- [Dependencies](#dependencies)
  - [Main Dependencies](#main-dependencies)
  - [Development Dependencies](#development-dependencies)
- [License](#license)
- [Contributing](#contributing)

## Project Status

**Note:** This project is currently under active development. The API, features, and benchmark results may change as development progresses. It is primarily intended for exploration and demonstration purposes at this stage.

## Platform Support

This project is currently being developed and tested on **Linux** and **macOS** systems. While Rust is cross-platform, specific SIMD intrinsics or performance characteristics might vary on other operating systems like Windows. The benchmarking and development efforts are focused on Unix-like environments.

## SIMD, Alignment, and Rust's `Vec<T>`/Slices

**SIMD (Single Instruction, Multiple Data)** allows a single instruction to operate on multiple data elements simultaneously. To do this efficiently, SIMD hardware often prefers or requires data to be loaded into its wide registers from memory locations that are **aligned** to specific boundaries (e.g., 16-byte, 32-byte, or 64-byte, depending on the SIMD instruction set like SSE, AVX, AVX-512).

- **Aligned vs. Unaligned Accesses**:

  - **Aligned accesses** are generally faster because the CPU can load/store the entire SIMD vector's worth of data in a single memory operation.
  - **Unaligned accesses** can incur performance penalties. The CPU might need multiple memory operations or internal fix-ups to handle data that crosses alignment boundaries. In some older or stricter SIMD instruction sets, unaligned accesses using aligned instructions could even cause a program to crash.

- **Rust's `Vec<T>` and Slices (`&[T]`, `&mut [T]`)**:

  - Standard Rust `Vec<T>` allocates memory with an alignment suitable for the type `T` itself. For example, a `Vec<f32>` will have its elements aligned to 4 bytes.
  - However, `Vec<T>` **does not** inherently guarantee that the _start_ of its data buffer will be aligned to larger boundaries (like 16 or 32 bytes) that are optimal for SIMD operations.
  - Slices created from a `Vec<T>` inherit this alignment characteristic.

- **Simdly's Approach**:

  - The intention for Simdly, particularly for its custom SIMD functions like `simd_add (simdly)`, is to operate directly on standard Rust `Vec<T>` and slices **without requiring or modifying their alignment**.
  - This means Simdly's SIMD code must be written to correctly handle potentially unaligned data. This typically involves:
    1.  Processing any initial unaligned elements at the beginning of a slice using scalar operations (or specific unaligned SIMD loads if careful) until a point where the remaining data is aligned relative to the SIMD vector width.
    2.  Performing the bulk of the operations using SIMD instructions on the aligned portion.
    3.  Processing any remaining unaligned elements at the end of the slice using scalar operations.
  - Alternatively, SIMD intrinsics that explicitly support unaligned loads and stores (e.g., `_mm_loadu_ps` in SSE, `_mm256_loadu_ps` in AVX) can be used throughout, offering safety at a potential performance cost compared to their aligned counterparts (`_mm_load_ps`, `_mm256_load_ps`).
  - The goal is to provide SIMD acceleration that is broadly applicable to common Rust data structures without imposing special alignment burdens on the user.

- **Benchmarking Context**:
  - The current benchmarks use `ndarray`. `ndarray` itself might employ strategies to manage or prefer aligned data for its internal operations, which could influence its performance characteristics.
  - The performance of `simd_add (simdly)` and `par_simd_add (avx2)` in the benchmarks reflects their execution on data as prepared by the benchmark setup (which uses `ndarray` for input generation). For a thorough understanding, specific benchmarks on data explicitly known to be unaligned (e.g., from `Vec`s with offset slices) would be beneficial.

## Features

- **SIMD Exploration**: Focuses on demonstrating and benchmarking SIMD-accelerated computations.
- **Parallel Processing**: Utilizes [Rayon](https://github.com/rayon-rs/rayon) for data parallelism, enabling efficient use of multi-core processors.
- **Numerical Computing**: Integrates with [Ndarray](https://github.com/rust-ndarray/ndarray) for n-dimensional array operations, common in scientific computing.
- **Comprehensive Benchmarking**: Employs [Criterion.rs](https://github.com/bheisler/criterion.rs) for detailed performance analysis of different vector addition strategies.
- **Optimized Release Builds**: `Cargo.toml` is configured for "fat" LTO and `codegen-units = 1` in release profiles for maximum performance.

## Getting Started

### Cloning the Repository

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/mtantaoui/simdly.git
    cd simdly
    ```

### Building the Project

Once you are in the project directory, you can build the project using Cargo:

```bash
cargo build
```

## Usage

Currently, Simdly primarily serves as a demonstration and benchmarking platform for SIMD and parallel computation techniques in Rust. The core logic for these demonstrations can be found within the benchmark files (e.g., `benches/add.rs`).

To use any library components (if developed in `src/`), you would typically add `simdly` as a dependency in your `Cargo.toml` and import its modules.

## Building

### Release Build

For a production-ready, optimized build:

```bash
cargo build --release
```

The project's `Cargo.toml` specifies aggressive optimization settings for release builds:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
```

This configuration aims to maximize runtime performance. The compiled artifacts will be located in `target/release/`.

## Benchmarking Environment

The performance results detailed in the "Benchmark Reports and Performance" section were obtained on the following system:

- **OS:** Ubuntu 22.04 jammy
- **CPU:** AMD EPYC 7571 @ 8x 2.2GHz
- **RAM:** 31828 MiB (Total)

Performance can vary significantly across different hardware and software configurations. The provided benchmark data should be considered relative to this specific environment.

## Running Benchmarks

### Overview

This project uses [Criterion.rs](https://criterion.rs) for robust statistical benchmarking. Benchmarks are crucial for evaluating the performance of SIMD implementations, parallelization strategies, and comparing them against standard library or other crate functionalities.

### Available Benchmarks

The `Cargo.toml` defines one benchmark suite:

- **`add`**:
  - **Source File**: `benches/add.rs` (assumed, based on `[[bench]] name = "add"`)
  - **Description**: This benchmark suite evaluates the performance of vector addition using various approaches. It is parameterized by vector size, allowing comparison across different scales. The benchmarked functions within this suite include:
    - `ndarray`: Vector addition using `ndarray`'s built-in operators.
    - `scalar_add`: A naive, element-by-element scalar addition.
    - `simd_add (simdly)`: A custom SIMD implementation for vector addition provided by this `simdly` project.
    - `par_simd_add (avx2)`: A parallelized SIMD implementation (likely using Rayon and targeting AVX2 instruction set).
  - **Harness**: `false` (Criterion's `main` macro is used directly in the benchmark source file).

### How to Run

You can run benchmarks using Cargo:

- **Run all benchmarks** (in this case, the `add` suite with all its parameterized groups):
  ```bash
  cargo bench
  ```
- **Run a specific benchmark group or function by name/filter**:
  Criterion allows filtering. For example, to run only tests related to "ndarray" for a specific size:
  ```bash
  cargo bench -- "VectorAddition/30000/ndarray"
  ```
  Or to run all benchmarks for the 30000 element vector size:
  ```bash
  cargo bench -- "VectorAddition/30000"
  ```

For more options, run with `--help`:

```bash
cargo bench -- --help
```

### Benchmark Reports and Performance

After running the benchmarks, Criterion generates detailed HTML reports. These reports can be found in the `target/criterion/` directory.

- **Summary Reports for each Parameterized Group**: Located at `target/criterion/VectorAddition_<size>/report/index.html` (e.g., `target/criterion/VectorAddition_30000/report/index.html`). These provide an overview and violin plots comparing all functions for that specific vector size.
- **Detailed Reports for each Function**: Located at `target/criterion/VectorAddition_<size>/<function_name>/report/index.html` (e.g., `target/criterion/VectorAddition_30000/ndarray/report/index.html`). These offer in-depth statistics, PDF plots, and regression analysis for a specific function at a particular size.

#### Summary of `add` Benchmark Results

The following tables summarize the performance (typical time per operation and throughput) for the `add` benchmark across different vector sizes and implementations, based on the Criterion report data generated on the [Benchmarking Environment](#benchmarking-environment) specified above. "Time" refers to the estimated time per iteration (Slope estimate from Criterion for linear sampling, or Mean for flat sampling).

**Vector Size: 30,000 elements** (`VectorAddition/30000`)

| Function              | Time (Mean) |
| :-------------------- | :---------- |
| `simd_add (simdly)`   | 6.1150 µs   |
| `scalar_add`          | 6.4880 µs   |
| `ndarray`             | 6.6200 µs   |
| `par_simd_add (avx2)` | 76.642 µs   |

**Vector Size: 150,000 elements** (`VectorAddition/150000`)

| Function              | Time (Mean) |
| :-------------------- | :---------- |
| `simd_add (simdly)`   | 32.194 µs   |
| `scalar_add`          | 35.082 µs   |
| `ndarray`             | 35.384 µs   |
| `par_simd_add (avx2)` | 187.10 µs   |

**Vector Size: 1,048,576 elements** (`VectorAddition/1048576`)

| Function              | Time (Mean) |
| :-------------------- | :---------- |
| `par_simd_add (avx2)` | 550.38 µs   |
| `simd_add (simdly)`   | 669.06 µs   |
| `scalar_add`          | 867.40 µs   |
| `ndarray`             | 886.10 µs   |

**Vector Size: 1,073,741,824 elements** (`VectorAddition/1073741824`)
_(Note: `ndarray` results were not present in the provided benchmark files for this size. Time is based on Mean as sampling mode was Flat.)_

| Function              | Time (Mean) |
| :-------------------- | :---------- |
| `par_simd_add (avx2)` | 1.4298 s    |
| `scalar_add`          | 4.9750 s    |
| `simd_add (simdly)`   | 4.1620 s    |
| `ndarray`             | 4.4635 s    |

**Observations from Benchmarks:**

- For smaller vector sizes (30k, 150k elements), the `simd_add (simdly)` implementation shows the best performance in terms of raw execution time per operation. The overhead of parallelism (`par_simd_add`) can make it slower for these smaller inputs.
- `ndarray` and `scalar_add` (likely auto-vectorized by the compiler) perform competitively at smaller sizes.
- As the vector size increases significantly (1M elements), `par_simd_add (avx2)` starts to show its strength due to parallel execution, outperforming the purely sequential SIMD and scalar versions. `simd_add (simdly)` is still faster than `scalar_add` and `ndarray`.
- For very large vectors (1B elements), `par_simd_add (avx2)` is substantially faster. The `simd_add (simdly)` implementation, being sequential, becomes much slower than the parallel version, and even slower than `scalar_add` in this specific run, which might indicate memory bandwidth limitations or other architectural effects at extreme scales for purely sequential processing.

These results highlight the trade-offs between different optimization strategies (explicit SIMD, parallelism, library abstractions) and how their effectiveness can vary with input size and the specific hardware capabilities (like AVX2 support and number of cores).

## Dependencies

Simdly relies on several high-quality Rust crates:

### Main Dependencies

- [`rayon`](https://crates.io/crates/rayon) (`1.10.0`): A data parallelism library for Rust, making it easy to convert sequential computations into parallel ones.

### Development Dependencies

These dependencies are used for development tasks like benchmarking:

- [`criterion`](https://crates.io/crates/criterion) (`0.6.0`): A statistics-driven benchmarking framework.
- [`ndarray`](https://crates.io/crates/ndarray) (`0.16.1`): An N-dimensional array-like (alternative to `Vec<Vec<T>>`) for Rust, essential for numerical computing.

For a full list of all direct and transitive dependencies, please refer to the `Cargo.lock` file.

## License

This project is licensed under the MIT License.
Copyright (c) 2025 Mahdi Tantaoui.

See the [LICENSE](LICENSE) file for the full license text.

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to:

1.  **Fork** the repository.
2.  Create a new **branch** for your feature or bug fix (e.g., `feature/my-new-feature` or `fix/issue-tracker-bug`).
3.  Make your changes. Ensure code is well-formatted using `cargo fmt`.
4.  Add or update relevant **tests** or **benchmarks** for your changes.
5.  Write clear and descriptive **commit messages**.
6.  Push your branch to your fork and open a **Pull Request** against the main repository.

Please consider opening an issue first to discuss any significant changes or new features you plan to implement.
