# Ceiling `ceil`

\\[
\lceil x \rceil = \min\\{n \in \mathbb{Z} : n \ge x\\}
\\]

**Exact** on every backend, like [`floor`](./floor.md). Implementation:
[`src/math/ceil.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/ceil.rs).

## 1. Special values

| Input            | Output  |
|------------------|---------|
| `−0.0`           | `−0.0`  |
| `−0.3`, `−0.7`   | `−0.0`  |
| `0.3`            | `1.0`   |
| `2.5`, `−2.5`    | `3.0`, `−2.0` |
| \\(\lvert x\rvert \ge 2^{23}\\) (`f32`), \\(2^{52}\\) (`f64`) | `x`: already an integer |
| `±∞`, NaN        | themselves |

## 2. One instruction per backend

| Backend  | Instruction                                                  |
|----------|--------------------------------------------------------------|
| AVX2     | `vroundps` / `vroundpd`, rounding mode "up" (`_mm256_ceil_ps`) |
| AVX-512  | `vrndscaleps` / `vrndscalepd`, mode "up", no precision exception |
| NEON     | `frintp`                                                     |

## 3. Scalar

\\(\lceil x \rceil = -\lfloor -x \rfloor\\), with `floor`'s float arithmetic
([`floor`, Sec. 3](./floor.md#3-scalar-rounding-with-float-arithmetic)).
The negations keep the sign of zero: \\(\lceil -0.7 \rceil = -\lfloor 0.7
\rfloor = -0.0\\), as `std` gives.

## 4. Tests

Every backend is compared with `std`'s `ceil` **bit for bit**, signed
zeros included: on every `f32` (an ignored test), and on 200 000 `f64`
samples plus values next to integers and halves.

## 5. Speed

The same as `floor`: one instruction per register, where `std` calls the
C library once per element on baseline `x86_64`. See
[Benchmarks](../benchmarks/README.md).
