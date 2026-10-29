# How simdly works

## Backends and dispatch

| Backend | ISA                 | `f32` lanes | `f64` lanes | Selected                         |
|---------|---------------------|-------------|-------------|----------------------------------|
| AVX-512 | AVX-512F            | 16          | 8           | at run time, on `x86_64`         |
| AVX2    | AVX2 + FMA          | 8           | 4           | at run time, on `x86_64`         |
| NEON    | NEON                | 4           | 2           | always, on `aarch64`             |
| scalar  | none                | 1           | 1           | when nothing else is available   |

On `x86_64`, each call checks the CPU's features once (the result is cached
by `std`) and runs the fastest backend available. Without `std`, only the
features enabled at compile time count. `simdly::backend()` reports the
backend in use.

## Registers and slices

Each backend has one register type per element type (`F32x8`, `F64x4`, …).
A slice is processed as full registers plus one tail: AVX2 and AVX-512 load
and store the tail with lane masks, NEON through a register-sized buffer.
Nothing past the end of a slice is read or written.

## One algorithm, every backend

A function like `acos` is written once, generic over a small `Float` trait:

| `Float` method      | AVX2                 | AVX-512                 | NEON          | scalar        |
|---------------------|----------------------|-------------------------|---------------|---------------|
| `a.mul_add(b, c)`   | `vfmadd` (fused)     | `vfmadd` (fused)        | `fmla` (fused)| `a * b + c`   |
| `select(m, a, b)`   | `vblendvps`          | masked blend            | `bsl`         | `if`          |
| `a.lt(b)`, …        | `vcmpps`             | `vcmpps` into a mask    | `fcmgt`, …    | `<`, …        |
| `a.and_bits(m)`     | `vpand`              | `vpandd`                | `and`         | `to_bits() &` |
| `a.sqrt()`, `a / b` | `vsqrtps`, `vdivps`  | `vsqrtps`, `vdivps`     | `fsqrt`, `fdiv` | `sqrt`, `/` |

Branches become selects: every lane computes each case of the algorithm
and keeps its own. The kernel compiles once per register type, to that
ISA's instructions; on scalar `f32` and `f64` it is ordinary code.

The scalar backend uses separate multiply and add, as `core` has no fused
multiply-add. Results can therefore differ between backends by a rounding,
so each function is tested against a precise reference on every backend,
not against another backend.

### Keeping the intrinsics inlined

Intrinsics only compile to single instructions inside code built with
their target features (`#[target_feature(enable = "avx2,fma")]`). Closures
and `impl Fn` arguments are compiled as separate functions *without* those
features: unless LLVM happens to inline them, every intrinsic inside
becomes a function call. Small operations got lucky; `acos` did not, and
ran 7× slower than scalar code until the slice drivers and kernels stopped
taking closures. simdly's operations are therefore types (`UnaryFn`,
`BinaryFn`) and `#[inline(always)]` functions, and the release binaries
contain no calls to intrinsics.
