# Striped SGEMM Approach for Column-Major Matrices: A Detailed Guide

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem with Naive SIMD](#the-problem-with-naive-simd)
3. [The Striped Loading Strategy](#the-striped-loading-strategy)
4. [Step-by-Step Breakdown](#step-by-step-breakdown)
5. [The De-striping Process](#the-de-striping-process)
6. [Performance Benefits](#performance-benefits)
7. [Implementation Details](#implementation-details)

## Introduction

Matrix multiplication (SGEMM: Single-precision General Matrix Multiply) is one of the most important operations in scientific computing. The challenge is efficiently computing `C = α·A·B + β·C` using SIMD (Single Instruction, Multiple Data) instructions while working with column-major matrices.

The **striped approach** is an advanced optimization technique that trades some computational complexity for significantly better SIMD utilization. This guide explains how this approach works specifically for column-major matrices.

## The Problem with Naive SIMD

### Traditional Approach Issues

In a naive SIMD implementation of matrix multiplication, you might try to:

1. Load 8 consecutive elements from matrix A
2. Load 8 consecutive elements from matrix B
3. Multiply them element-wise
4. Store the results

**Problem**: This doesn't compute the correct matrix multiplication! Matrix multiplication requires computing dot products between rows of A and columns of B, not element-wise multiplication.

### Memory Layout Challenges

With column-major matrices:

- **Matrix A (8×k)**: Elements in each column are contiguous in memory
- **Matrix B (k×8)**: Elements in each row are scattered across memory (stride = k)
- **Matrix C (8×8)**: We want to compute 8 columns, each with 8 elements

The challenge is efficiently computing all 64 elements of the result matrix C using 256-bit AVX vectors that can hold 8 float32 values.

## The Striped Loading Strategy

### Core Concept

Instead of trying to compute matrix elements in their natural order, the striped approach:

1. **Loads data in patterns optimized for SIMD**
2. **Computes results in a "striped" (interleaved) order**
3. **Uses permutations to rearrange results into correct matrix order**

### Why "Striped"?

The name comes from how the data is arranged after loading. Instead of having consecutive matrix elements, we have elements from different rows/columns interleaved (striped) together.

## Step-by-Step Breakdown

### Step 1: Loading Matrix A (Striped Pattern)

For each column of matrix A, we create special patterns:

```rust
// Load column p of A: [a0, a1, a2, a3, a4, a5, a6, a7]
let a_col = _mm256_loadu_ps(a.add(p * 8));

// Create striped patterns:
let a_even = _mm256_moveldup_ps(a_col);      // [a0, a0, a2, a2, a4, a4, a6, a6]
let a_odd = _mm256_movehdup_ps(a_col);       // [a1, a1, a3, a3, a5, a5, a7, a7]
```

**Key insight**: Each element is duplicated, creating patterns that will be useful for computing multiple matrix elements simultaneously.

### Step 2: Creating Permuted Patterns

We create additional patterns by permuting the striped data:

```rust
// Permute even pattern: [a2, a2, a0, a0, a6, a6, a4, a4]
let a_even_perm = _mm256_permute_ps(a_even, 0b01001110);

// Permute odd pattern: [a3, a3, a1, a1, a7, a7, a5, a5]
let a_odd_perm = _mm256_permute_ps(a_odd, 0b01001110);
```

Now we have 4 different patterns from the same column of A.

### Step 3: Loading Matrix B

```rust
// Load row p of B: [b0, b1, b2, b3, b4, b5, b6, b7]
let b_row = _mm256_loadu_ps(b.add(p * 8));

// Create permuted B by swapping 128-bit lanes: [b4, b5, b6, b7, b0, b1, b2, b3]
let b_perm = _mm256_permute2f128_ps(b_row, b_row, 0x01);
```

### Step 4: Computing Striped Outer Products

We compute 8 different combinations using FMA (Fused Multiply-Add):

```rust
ab_striped[0] = _mm256_fmadd_ps(a_even, b_row, ab_striped[0]);
ab_striped[1] = _mm256_fmadd_ps(a_even_perm, b_row, ab_striped[1]);
ab_striped[2] = _mm256_fmadd_ps(a_even, b_perm, ab_striped[2]);
ab_striped[3] = _mm256_fmadd_ps(a_even_perm, b_perm, ab_striped[3]);

ab_striped[4] = _mm256_fmadd_ps(a_odd, b_row, ab_striped[4]);
ab_striped[5] = _mm256_fmadd_ps(a_odd_perm, b_row, ab_striped[5]);
ab_striped[6] = _mm256_fmadd_ps(a_odd, b_perm, ab_striped[6]);
ab_striped[7] = _mm256_fmadd_ps(a_odd_perm, b_perm, ab_striped[7]);
```

### Step 5: Understanding the Striped Result

After all iterations, each `ab_striped[i]` vector contains 8 elements, but they're not in natural matrix order. Instead, they're distributed in a specific pattern that comes from the way we duplicated and permuted the input data.

**Example**: `ab_striped[0]` might contain elements like `[C₀₀, C₀₁, C₂₂, C₂₃, C₄₄, C₄₅, C₆₆, C₆₇]` where `Cᵢⱼ` represents the element at row i, column j of the result matrix.

## The De-striping Process

### Why De-striping is Necessary

The 8 vectors `ab_striped[0..7]` contain all 64 elements of the result matrix, but they're scrambled. We need to rearrange them so that:

- `result[0]` contains column 0: `[C₀₀, C₁₀, C₂₀, C₃₀, C₄₀, C₅₀, C₆₀, C₇₀]`
- `result[1]` contains column 1: `[C₀₁, C₁₁, C₂₁, C₃₁, C₄₁, C₅₁, C₆₁, C₇₁]`
- etc.

### De-striping Algorithm

#### Phase 1: Shuffle Within Vectors

```rust
let shuffle_mask = 0b11100100; // 3,2,1,0

// Combine pairs of striped vectors
let pair01 = _mm256_shuffle_ps(striped[0], striped[1], shuffle_mask);
let pair23 = _mm256_shuffle_ps(striped[1], striped[0], shuffle_mask);
// ... continue for all pairs
```

**Purpose**: Group related elements from different striped vectors.

#### Phase 2: Permute 128-bit Lanes

```rust
// Combine pairs using lane permutation to form final columns
let col0 = _mm256_permute2f128_ps(pair01, pair45, 0x20);
let col4 = _mm256_permute2f128_ps(pair01, pair45, 0x31);
// ... continue for all columns
```

**Purpose**: Rearrange the 128-bit lanes to put each complete column together.

### Visual Example of De-striping

**Before de-striping** (striped vectors):

```
striped[0]: [C₀₀, C₀₁, C₂₂, C₂₃, C₄₄, C₄₅, C₆₆, C₆₇]
striped[1]: [C₂₀, C₂₁, C₀₂, C₀₃, C₆₄, C₆₅, C₄₆, C₄₇]
striped[2]: [C₀₄, C₀₅, C₂₆, C₂₇, C₄₀, C₄₁, C₆₂, C₆₃]
...
```

**After de-striping** (column-major result):

```
result[0]: [C₀₀, C₁₀, C₂₀, C₃₀, C₄₀, C₅₀, C₆₀, C₇₀]  // Column 0
result[1]: [C₀₁, C₁₁, C₂₁, C₃₁, C₄₁, C₅₁, C₆₁, C₇₁]  // Column 1
result[2]: [C₀₂, C₁₂, C₂₂, C₃₂, C₄₂, C₅₂, C₆₂, C₇₂]  // Column 2
...
```

## Performance Benefits

### 1. Maximized SIMD Utilization

- **8 parallel operations** per SIMD instruction
- **Efficient use of FMA** (Fused Multiply-Add) instructions
- **Minimal scalar operations** in the inner loop

### 2. Improved Memory Access Patterns

- **Sequential loads** where possible (column-major A)
- **Reduced gather operations** compared to naive approaches
- **Better cache utilization**

### 3. Instruction-Level Parallelism

- **Overlapped computation and memory access**
- **Efficient pipeline utilization**
- **Reduced dependency chains**

### Performance Trade-offs

**Pros**:

- 3-5x speedup over naive implementations
- Excellent SIMD utilization (close to peak theoretical performance)
- Scales well with larger matrices

**Cons**:

- Increased code complexity
- Additional permutation overhead
- More registers required
- Harder to debug and maintain

## Implementation Details

### Register Pressure

The algorithm uses:

- **8 registers** for striped accumulation (`ab_striped[0..7]`)
- **4 registers** for A patterns (`a_even`, `a_odd`, `a_even_perm`, `a_odd_perm`)
- **2 registers** for B patterns (`b_row`, `b_perm`)
- **Additional registers** for de-striping temporaries

Total: ~16-20 AVX registers (out of 16 available), requiring careful register allocation.

### Memory Alignment

For optimal performance:

- **32-byte alignment** for matrix data when possible
- **Use `_mm256_load_ps`** instead of `_mm256_loadu_ps` for aligned data
- **Consider padding** matrices to maintain alignment

### Loop Unrolling

Production implementations often unroll the k-loop by 2-4x to:

- **Reduce loop overhead**
- **Increase instruction-level parallelism**
- **Better utilize CPU resources**

### Prefetching

Add software prefetching for larger matrices:

```rust
// Prefetch next iteration's data
_mm_prefetch(a.add((p + 1) * 8) as *const i8, _MM_HINT_T0);
_mm_prefetch(b.add((p + 1) * 8) as *const i8, _MM_HINT_T0);
```

## Conclusion

The striped approach to SGEMM represents a sophisticated optimization technique that:

1. **Transforms the computation** from natural matrix order to SIMD-friendly patterns
2. **Maximizes parallel execution** using all available SIMD lanes
3. **Uses permutations** to efficiently rearrange data into the correct output format

While more complex than naive approaches, this technique is essential for achieving high-performance matrix multiplication on modern CPUs. The same principles extend to larger kernels (16×16, 32×32) and different data types (double precision, integer), making it a fundamental technique in optimized BLAS libraries.

The key insight is that **computational efficiency often requires trading simplicity for performance**, and the striped approach exemplifies this trade-off in the context of dense linear algebra operations.
