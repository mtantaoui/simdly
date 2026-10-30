# Four-dimensional hypotenuse `hypot4`

\\[
\operatorname{hypot4}(x, y, z, w) = \sqrt{x^2 + y^2 + z^2 + w^2}
\\]

without overflow or underflow in between: the norm of a quaternion or a
4-vector. Implementation:
[`src/math/hypot.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/hypot.rs).
Error: **0.500 ULP** for normal results and **0.75 ULP** for subnormal
ones, `f32` and `f64`, on every backend.

## 1. Same code as `hypot3`

`hypot3` and `hypot4` are one kernel, generic over the number of inputs
\\(N\\) and unrolled at compile time
([`hypot3`, Sec. 2](./hypot3.md#2-exact-sum-of-squares)):

1. Scale every input by the power of 2 chosen from the largest.
2. Split each square exactly, \\(x^2 = h_x + \ell_x\\).
3. Sum the high parts exactly with a chain of \\(N - 1\\) `two_sum`s:
   the largest square can be \\(1/N\\) of the sum, so the Newton step's
   residual is taken against the exact sum \\(t\\), not against the
   largest square.
4. \\(r = \sqrt{t + \ell}\\), one Newton step, scale back.

Special values are as for `hypot3`: any `±∞` gives `+∞`, even with a NaN
among the others; otherwise NaN gives NaN; all zeros give `+0`.

## 2. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against a
double-double reference with four squares:

| Element | Vectors                                                      | Normal results | Subnormal results |
|---------|--------------------------------------------------------------|----------------|-------------------|
| `f32`   | 10⁷: every magnitude, nearby exponents, subnormal vectors    | 0.500          | 0.750             |
| `f64`   | 10⁶, the same families                                       | 0.500          | 0.749             |

Fused (SIMD) and Dekker (scalar) give the same maxima. The tests run
200 000 vectors per type on every backend, and 10⁸ more `f32` vectors in
an ignored test (`cargo test --release -- --ignored many_vectors`).

## 3. Cost

One more square and `two_sum` than `hypot3`. `std`'s overflow-safe
equivalent, three nested `hypot`s, is 4.3× to 4.8× slower for `f32` and
3.9× to 6.1× for `f64` on AVX2 (see [Benchmarks](../benchmarks/README.md)).
