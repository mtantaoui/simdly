# Inverse hyperbolic function `atanh`

\\[
\operatorname{atanh} : (-1, 1) \to \mathbb{R}, \qquad
\operatorname{atanh} x = \frac12 \ln\frac{1 + x}{1 - x}
\\]

simdly builds `atanh` on [`log1p`](./log1p.md)'s reduction and series,
with the same code for `f32` and `f64`. Implementation:
[`src/math/inverse_hyperbolic.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/inverse_hyperbolic.rs).
Error: **0.853 ULP** (`f32`) and **0.845 ULP** (`f64`) at most, on every
backend ([measurements](#4-measured-error)).

## 1. Special values

| Input                                   | Output            |
|-----------------------------------------|-------------------|
| `±0`                                    | `±0`              |
| tiny \\(x\\)                            | \\(x\\), subnormals included |
| `±1`                                    | `±∞`              |
| \\(\lvert x\rvert > 1\\), `±∞`          | NaN               |
| NaN                                     | NaN               |

## 2. The formula, and why not as it stands

`atanh` is odd; with \\(t = \lvert x\rvert\\),

\\[
\operatorname{atanh} t = \tfrac12 \operatorname{log1p}(q), \qquad q = \frac{2t}{1 - t}
\\]

which avoids the cancellation of \\(\ln(1 + t) - \ln(1 - t)\\) near 0.
But rounding \\(q\\) and then calling `log1p` rounds twice: \\(q\\)'s error
passes into the result. Built from simdly's own `log1p`, that reaches
**2.01 ULP** in `f32`, measured over every input.

## 3. `q` to twice the precision, one logarithm

- **\\(1 - t = d_h + d_l\\) exactly** (`two_sum`): \\(d_l\\) is nonzero
  for \\(t < 0.5\\), where the subtraction rounds.
- **\\(q = q_h + q_l\\):** \\(q_h = 2t \cdot (1/d_h)\\), then one
  correction step,
  \\[
  q_l = \bigl((2t - q_h d_h) - q_h d_l\bigr) \cdot \frac{1}{d_h}
  \\]
  where \\(2t - q_h d_h\\) is exact (`residual`: a fused multiply-add, or
  Dekker's product on the scalar backend). The one division serves both.
- **\\(\ln(1 + q_h + q_l)\\)** as in `log1p`: \\(1 + q_h = u + c\\)
  exactly, \\(u = m \cdot 2^n\\) reduced as in [`ln`](./ln.md). For
  \\(n = 0\\) the series takes \\(q_h\\) itself, with \\(q_l/u\\) as the
  correction; otherwise \\(f = m - 1\\), exact, with \\((c + q_l)/u\\). The
  correction joins the smallest terms of the reconstruction.
- **The halving is exact**: the result is never subnormal (below).

Below \\(2^{-12}\\) (`f32`) and \\(2^{-28}\\) (`f64`),
\\(\operatorname{atanh} x = x(1 + x^2/3 + \dots)\\) rounds to \\(x\\),
which is returned. The kernel's input is capped there: for smaller
\\(t\\), the series' \\(s^4\\) would turn subnormal in `f32`: subnormal
arithmetic made [`softplus`](./softplus.md) 2.3× slower on x86.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                                 | Fused (SIMD) | Unfused (scalar) |
|---------|------------------------------------------------------------------------|--------------|------------------|
| `f32`   | every input in \\((-1, 1)\\)                                           | 0.822        | 0.853            |
| `f64`   | 32 million samples: \\((-1, 1)\\), \\([0.1, 0.6]\\), and \\(\lvert x\rvert\\) from \\(2^{-30}\\) to 1 by magnitude, double-double reference | 0.826 | 0.845 |

The maxima sit near \\(x \approx 0.17\\), where \\(1 + q\\) crosses
\\(\sqrt2\\) and the series switches between \\(q_h\\) and \\(m - 1\\), as
`log1p`'s sit at the ends of its \\(n = 0\\) range. The `f64` reference,
`atanh_dd`, divides in double-double and takes [`log1p`](./log1p.md)'s
double-double reference of the two-part \\(q\\).

## 5. Cost

One division for \\(1/d_h\\), `two_sum` twice, then `log1p`'s reduction
and series with its division. On AVX2 at 4096 elements in
\\([-0.99, 0.99]\\), `atanh` is 3.3× faster than `std`'s for `f32` and
1.8× for `f64`; near 0, where `std`'s `f64` `atanh` takes a shorter path,
1.2× (see [Benchmarks](../benchmarks/README.md)).
