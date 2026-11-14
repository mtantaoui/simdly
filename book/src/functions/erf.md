# Error function `erf`

\\[
\operatorname{erf} : \mathbb{R} \to (-1, 1), \qquad
\operatorname{erf} x = \frac{2}{\sqrt\pi} \int_0^x e^{-t^2}\, dt
\\]

simdly follows fdlibm's `s_erf.c` (musl's `erf.c`), with the same code for
`f32` and `f64`; `f32` uses the same coefficients rounded, as musl's
`erff.c`. Implementation:
[`src/math/erf.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/erf.rs).
Error: **0.896 ULP** (`f32`) and **0.867 ULP** (`f64`) at most, on every
backend, subnormal results included ([measurements](#4-measured-error)).

## 1. Special values and range

| Input                                   | Output            |
|-----------------------------------------|-------------------|
| `±0`                                    | `±0`              |
| tiny \\(x\\)                            | \\(\frac{2}{\sqrt\pi} x\\), subnormals included |
| `±∞`                                    | `±1`              |
| \\(\lvert x\rvert\\) above 3.92 (`f32`), 5.92 (`f64`) | `±1` |
| NaN                                     | NaN               |

## 2. fdlibm's four ranges

With \\(a = \lvert x\rvert\\) (`erf` is odd):

| Range                  | Formula                                                    | Approximation error |
|------------------------|------------------------------------------------------------|---------------------|
| \\(a < 0.84375\\)      | \\(x + x \cdot P(x^2)/Q(x^2)\\)                              | \\(2^{-57.90}\\)    |
| \\(a < 1.25\\)         | \\(\mathrm{erx} + P(a - 1)/Q(a - 1)\\), \\(\mathrm{erx} \approx \operatorname{erf} 1\\) | \\(2^{-59.06}\\) |
| \\(a < 1/0.35\\), \\(a < 6\\) | \\(1 - e^{-a^2 - 0.5625 + R(1/a^2)/S(1/a^2)}/a\\)     | \\(2^{-62.57}\\), \\(2^{-61.52}\\) |
| \\(a \ge 6\\)          | \\(1\\)                                                    |                     |

The rational functions below 1.25 share one evaluation, and so do those
above, their coefficients selected per lane; each half runs only when some
lane of the register needs it, and below 1.25 the second coefficient set
is selected only when some lane is in \\([0.84375, 1.25)\\).

**The exponent.** fdlibm truncates \\(a\\) so that \\(a^2\\) is exact and
computes two `exp`s. Here \\(a^2 = s_h + s_l\\) exactly (`residual`: a
fused multiply-add, or Dekker's product on the scalar backend), the
exponent is a two-part sum \\(e_h + e_l\\) (`two_sum`), and one
`exp_parts` (see [`sigmoid`](./sigmoid.md)) takes \\(e_h\\), the low part
multiplying in: \\(e^{e_h + e_l} = (y_h + y_l)(1 + e_l) \cdot 2^k\\). An
error in \\(1 - \operatorname{erfc}\\) counts only in proportion to
\\(\operatorname{erfc} a < 0.08\\).

## 3. Below 0.84375: where the formula as it stands falls short

Evaluated as written, \\(x + x \cdot P/Q\\) reaches **0.99 ULP** in `f64`
on the scalar backend, measured: \\(P/Q \approx 0.128\\) rounds just above
0.125, where its ULP is largest relative to it, and so do
\\(Q \approx 1\\), the quotient, and the unfused product \\(x \cdot P/Q\\).
So the range is split at \\(0.40625\\), about where the two forms below
are equally good:

- **Below 0.40625**, fdlibm's \\(P/Q = p_0 + z \cdot P'(z)/Q(z)\\) exactly,
  \\(z = x^2\\), \\(P'_i = p_{i+1} - p_0 q_{i+1}\\), so
  \\[
  \operatorname{erf} x = x \cdot \tfrac{2}{\sqrt\pi} + x \cdot z \cdot P'(z)/Q(z)
  \\]
  with \\(2/\sqrt\pi\\) as two floats and \\(x \cdot (2/\sqrt\pi)_h\\)
  exact (`residual`): only the small second term rounds before the final
  sum. Below \\(2^{-28}\\), \\(z\\) is taken as 0, and \\(x\\) is scaled by
  \\(2^{128}\\) (`f64`) or \\(2^{64}\\) (`f32`) first and back last, so the
  product stays exact and a subnormal result rounds once.
- **From 0.40625**, where \\(z \cdot P'/Q\\) is no longer small,
  \\(x + x \cdot P/Q\\) itself, the product exact and the sum rounded once
  (`two_sum`), fused or not.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                       | Fused (SIMD) | Unfused (scalar) |
|---------|--------------------------------------------------------------|--------------|------------------|
| `f32`   | every finite input, against `libm`'s `f64` `erf`             | 0.862        | 0.896            |
| `f64`   | 72 million samples: \\(\lvert x\rvert\\) from \\(2^{-1074}\\) to 0.84 by magnitude, \\([-6.5, 6.5]\\), \\([0.3, 1.4]\\), \\([1.2, 3]\\), double-double reference | 0.830 | 0.867 |

The maxima sit near the top of the first range, \\(x \approx 0.80\\) to
0.84, where \\(x \cdot P/Q\\) is largest against the result. The `f64`
reference, `erf_dd`, sums the series
\\(\frac{2}{\sqrt\pi} e^{-a^2} \sum_n \frac{2^n a^{2n+1}}{1 \cdot 3 \cdots (2n+1)}\\),
whose terms are all positive, in double-double, with \\(e^{-a^2}\\) from
[`exp`](./exp.md)'s double-double reference; tiny results are kept scaled
by \\(2^{200}\\), so subnormal results are measured against all their
bits.

## 5. Cost

Below 1.25: one division and two rational functions of degree 6, with the
exact products and sums above. From 1.25: the same, plus `exp_parts` and
the reciprocal \\(1/a\\). On AVX2 at 4096 elements, compared with `libm`'s
scalar `erf` (`std` has none):

| Inputs          | `f32` | `f64` |
|-----------------|-------|-------|
| \\([-0.8, 0.8]\\) | 2.0×  | 1.0×  |
| \\([-3, 3]\\)     | 2.5×  | 1.6×  |
| \\([-6, 6]\\)     | 3.0×  | 2.0×  |

On small inputs, `libm`'s `f64` `erf` evaluates one rational function per
element; the SIMD kernel computes both forms below 0.84375 in every lane,
and only matches it there (see [Benchmarks](../benchmarks/README.md)).
