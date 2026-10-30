# Three-dimensional hypotenuse `hypot3`

\\[
\operatorname{hypot3}(x, y, z) = \sqrt{x^2 + y^2 + z^2}
\\]

without overflow or underflow in between, as [`hypot`](./hypot.md).
Implementation:
[`src/math/hypot.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/hypot.rs).
Error: **0.500 ULP** for normal results and **0.75 ULP** for subnormal
ones, `f32` and `f64`, on every backend ([measurements](#4-measured-error)).

## 1. Special values

As `hypot`: the result does not depend on the signs.

| `x`, `y`, `z`                  | Result      |
|--------------------------------|-------------|
| any `±∞`                       | `+∞`, even if another is NaN |
| any NaN, none infinite         | NaN         |
| all `±0`                       | `+0`        |

## 2. Exact sum of squares

Each input is scaled by the power of 2 chosen from the **largest** one
([`hypot`, Sec. 5](./hypot.md#5-scaling)), and each square split exactly,
\\(x^2 = h_x + \ell_x\\) ([`hypot`, Sec. 3](./hypot.md#3-exact-squares)).

`hypot`'s Newton step computes \\(h_x - r^2\\) with `residual`, exact on
the scalar backend (Dekker's product) only when \\(h_x \ge r^2 / 2\\). With
three squares, the largest can be a third of the sum. So the three high
parts are first summed **exactly** with two `two_sum`s:

\\[
h_x + h_y + h_z = t + e
\\]

and the Newton step uses \\(t \approx r^2\\), for which the residual is
exact on every backend:

\\[
r = \sqrt{t + \ell}, \qquad r' = r + \frac{(t - r^2) + \ell}{2r}, \qquad \ell = (\ell_x + \ell_y + \ell_z) + e
\\]

The sum being exact in any order, the inputs need no sorting: only the
largest is found, for the scale. Without the exact sum, the scalar backend
measured 0.975 ULP.

## 3. Subnormal results

As for `hypot`, they are rounded once in the scaled domain and again when
scaled back: 0.75 ULP.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against
`hypot`'s double-double reference with three squares:

| Element | Triples                                                       | Normal results | Subnormal results |
|---------|---------------------------------------------------------------|----------------|-------------------|
| `f32`   | 10⁷: every magnitude, nearby exponents, subnormal triples     | 0.500          | 0.750             |
| `f64`   | 10⁶, the same families                                        | 0.500          | 0.749             |

Fused (SIMD) and Dekker (scalar) give the same maxima. The tests run
200 000 triples per type on every backend, and 10⁸ more `f32` triples in
an ignored test (`cargo test --release -- --ignored many_triples`).

## 5. Cost

Per register: three products and three `residual`s for the squares, two
`two_sum`s, a square root, one more `residual`, one division, and the
selects. `std` has no `hypot3`; its overflow-safe equivalent,
`x.hypot(y).hypot(z)`, is 2.4× to 4.3× slower on AVX2 (see
[Benchmarks](../benchmarks/README.md)).
