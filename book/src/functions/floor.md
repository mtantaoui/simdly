# Floor `floor`

\\[
\lfloor x \rfloor = \max\\{n \in \mathbb{Z} : n \le x\\}
\\]

**Exact** on every backend: the result is an integer that is always a
float, so there is nothing to approximate. Implementation:
[`src/math/floor.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/floor.rs).

## 1. Special values

| Input            | Output  |
|------------------|---------|
| `−0.0`           | `−0.0`  |
| `−0.3`           | `−1.0`  |
| `2.5`, `−2.5`    | `2.0`, `−3.0` |
| \\(\lvert x\rvert \ge 2^{23}\\) (`f32`), \\(2^{52}\\) (`f64`) | `x`: already an integer |
| `±∞`, NaN        | themselves |

## 2. One instruction per backend

| Backend  | Instruction                                                    |
|----------|----------------------------------------------------------------|
| AVX2     | `vroundps` / `vroundpd`, rounding mode "down" (`_mm256_floor_ps`) |
| AVX-512  | `vrndscaleps` / `vrndscalepd`, mode "down", no precision exception |
| NEON     | `frintm`                                                       |

## 3. Scalar: rounding with float arithmetic

`core` has no `floor` without `std`, and on baseline `x86_64` (no
SSE4.1) `std`'s `floor` is a call into the C library. The scalar backend
uses three float operations instead:

1. \\(r = (\lvert x\rvert + 2^{23}) - 2^{23}\\): the sum lands where the ULP
   is 1, so it rounds \\(\lvert x\rvert\\) to the nearest integer, and the
   subtraction is exact. (\\(2^{52}\\) for `f64`.)
2. Give \\(r\\) the sign of \\(x\\), from its sign bit, so `−0.3` gives
   `−0.0` rather than `+0.0`.
3. If \\(r > x\\), rounding went up: \\(r - 1\\).

From \\(2^{23}\\) on, every float is an integer and is returned as it is,
like `±∞` and NaN.

## 4. Tests

Every backend is compared with `std`'s `floor` **bit for bit**, signed
zeros included: on every `f32` (an ignored test, 30 seconds), and on
200 000 `f64` samples plus values next to integers and halves, where
rounding to nearest and rounding down differ.

`floor` is also part of the `Float` operations every kernel can use, for
the range reductions of `exp` and the trigonometric functions.

## 5. Speed

On AVX2, `floor` is 6× to 25× faster than `std` for `f32`: `std` calls
the C library once per element, simdly rounds 8 lanes in one instruction.
See [Benchmarks](../benchmarks/README.md).
