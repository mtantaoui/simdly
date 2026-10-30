# Hypotenuse `hypot`

\\[
\operatorname{hypot}(x, y) = \sqrt{x^2 + y^2}
\\]

without overflow or underflow in between: `hypot(1e30, 1e30)` is
\\(1.4 \cdot 10^{30}\\) in `f32`, although \\(x^2\\) overflows.
Implementation:
[`src/math/hypot.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/hypot.rs).
Error: **0.666 ULP** (`f32`) and **0.637 ULP** (`f64`) at most for normal
results, **0.83 ULP** for subnormal ones, on every backend
([measurements](#6-measured-error)).

## 1. Special values

As C99 Annex F and Rust's `f32::hypot`:

| `x`, `y`                       | Result      |
|--------------------------------|-------------|
| either `±∞`                    | `+∞`, even if the other is NaN |
| either NaN, neither infinite   | NaN         |
| `(±0, ±0)`                     | `+0`        |
| `(x, ±0)`                      | \\(\lvert x\rvert\\) |

The result does not depend on the signs.

## 2. Three roundings

Computed directly, \\(\sqrt{x^2 + y^2}\\) rounds \\(x^2\\), \\(y^2\\), the
sum and the square root. The square root halves the relative error of its
argument, but the total can still approach 1 ULP. musl's `hypotf` sidesteps
this by computing in `f64`; SIMD lanes cannot widen cheaply, so simdly
removes the roundings instead.

## 3. Exact squares

As musl's `f64` `hypot`, each square is split into two floats:

\\[
x^2 = h_x + \ell_x, \qquad h_x = \operatorname{round}(x^2), \quad \ell_x = x^2 - h_x
\\]

\\(\ell_x\\) is exactly representable, and computed exactly by the `Float`
operation `residual` from [`atan2`](./atan2.md#3-no-rounded-quotient): a
fused multiply-add on SIMD, Dekker's product on scalar. The four parts are
summed smallest first and the square root taken:

\\[
r = \sqrt{((\ell_y + \ell_x) + h_y) + h_x}
\\]

## 4. One Newton step

That alone measures **0.996 ULP** in `f32`: within the bound, without
margin. One Newton step for the square root,

\\[
r' = r + \frac{(x^2 + y^2) - r^2}{2r}, \qquad (x^2 + y^2) - r^2 = (h_x - r^2) + h_y + (\ell_x + \ell_y)
\\]

with \\(h_x - r^2\\) computed exactly by `residual` again, brings it to
**0.666 ULP**. It is not optional: without it, 10⁸ random `f32` pairs find
an input at 1.004 ULP.

## 5. Scaling

- **Overflow and underflow.** Both inputs are multiplied by a power of 2
  chosen from the **larger** one, so that its square and that square's low
  part are normal floats: by \\(2^{-80}\\) above \\(2^{60}\\), by
  \\(2^{100}\\) below \\(2^{-40}\\) (`f64`: \\(2^{-600}\\) above
  \\(2^{500}\\), \\(2^{600}\\) below \\(2^{-450}\\)). The smaller square
  may lose bits to underflow, but those bits are far below the result's
  ULP.
- **Very different magnitudes.** When the smaller input is below
  \\(2^{-26}\\) (`f64`: \\(2^{-55}\\)) times the larger, the result is the
  larger one: \\(\sqrt{1 + \epsilon^2}\\) rounds to 1.
- **Scale back once.** The Newton correction is added in the scaled domain
  and the sum multiplied back once. Scaling \\(r\\) and the correction
  separately, `r·s + c·s` in one fused multiply-add, lets the correction
  underflow on its own for subnormal results: that measured 1.054 ULP.

For subnormal results, the scaled result rounds once to the full
significand and again when scaled back to fewer bits: hence their 0.83
ULP bound instead of 0.67.

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against a
reference that sums the squares in double-double arithmetic and takes one
Newton step, errors measured in the scaled domain so that subnormal results
are exact too:

| Element | Pairs                                                         | Normal results | Subnormal results |
|---------|---------------------------------------------------------------|----------------|-------------------|
| `f32`   | 3·10⁷: every magnitude, nearby exponents, subnormal pairs     | 0.666          | 0.827             |
| `f64`   | 3·10⁵, the same families, 300-bit reference                   | 0.637          | 0.807             |

Fused (SIMD) and Dekker (scalar) give the same maxima. The tests run
200 000 pairs per type on every backend, and 10⁸ more `f32` pairs in an
ignored test (`cargo test --release -- --ignored many_pairs`).

## 7. Cost

Per register: two comparisons and selects for the scaling, four products
(two of them exact via `residual`), a square root, one more `residual`,
one division, and the special-value selects. On AVX2, `hypot` is about
2.2× faster than `std` for `f32` and 2× for `f64` (see
[Benchmarks](../benchmarks/README.md)).
