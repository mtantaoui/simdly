# Inverse hyperbolic functions `asinh`, `acosh` and `atanh`

\\[
\operatorname{asinh} x = \ln\bigl(x + \sqrt{x^2 + 1}\bigr), \qquad
\operatorname{acosh} x = \ln\bigl(x + \sqrt{x^2 - 1}\bigr), \qquad
\operatorname{atanh} x = \frac12 \ln\frac{1 + x}{1 - x}
\\]

simdly builds all three on [`log1p`](./log1p.md)'s reduction and series,
with the same code for `f32` and `f64`. Implementation:
[`src/math/inverse_hyperbolic.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/inverse_hyperbolic.rs).
Error at most, on every backend ([measurements](#5-measured-error)):

| Function | `f32`         | `f64`         |
|----------|---------------|---------------|
| `asinh`  | **0.846 ULP** | **0.858 ULP** |
| `acosh`  | **0.841 ULP** | **0.851 ULP** |
| `atanh`  | **0.853 ULP** | **0.845 ULP** |

## 1. Special values

| Input                                   | `asinh`      | `acosh`      | `atanh`      |
|-----------------------------------------|--------------|--------------|--------------|
| `±0`                                    | `±0`         | NaN          | `±0`         |
| tiny \\(x\\)                            | \\(x\\), subnormals included | NaN | \\(x\\), subnormals included |
| `1`                                     | \\(\ln(1 + \sqrt2)\\) | `+0` | `+∞`        |
| `−1`                                    | \\(-\ln(1 + \sqrt2)\\) | NaN | `−∞`        |
| \\(x < 1\\)                             |              | NaN          |              |
| \\(\lvert x\rvert > 1\\)                |              |              | NaN          |
| `+∞`                                    | `+∞`         | `+∞`         | NaN          |
| `−∞`                                    | `−∞`         | NaN          | NaN          |
| NaN                                     | NaN          | NaN          | NaN          |

## 2. One logarithm of an argument formed exactly

Each function is \\(\ln(1 + p)\\) for some \\(p\\): `asinh` and `atanh` are
odd, so with \\(t = \lvert x\rvert\\),

\\[
\operatorname{asinh} t = \operatorname{log1p}\bigl(t + (\sqrt{1 + t^2} - 1)\bigr), \quad
\operatorname{acosh} x = \operatorname{log1p}\bigl(t + \sqrt{t^2 + 2t}\bigr),\ t = x - 1, \quad
\operatorname{atanh} t = \tfrac12 \operatorname{log1p}\Bigl(\frac{2t}{1 - t}\Bigr)
\\]

which avoid the cancellation of the textbook forms: \\(x + \sqrt{x^2 + 1}\\)
for \\(x < 0\\), \\(x^2 - 1\\) near 1, \\(\ln(1 + t) - \ln(1 - t)\\) near
0. But rounding \\(p\\) and then calling `log1p` rounds twice: \\(p\\)'s
error passes into the result. For `atanh`, built from simdly's own `log1p`,
that reaches **2.01 ULP** in `f32`, measured over every input.

So \\(p = p_h + p_l\\) is formed to about twice the element type's
precision, and \\(\ln(1 + p_h + p_l)\\) proceeds as in `log1p`:
\\(1 + p_h = u + c\\) exactly, \\(u = m \cdot 2^n\\) reduced as in
[`ln`](./ln.md). For \\(n = 0\\) the series takes \\(p_h\\) itself, with
\\(p_l/u\\) as the correction; otherwise \\(f = m - 1\\), exact, with
\\((c + p_l)/u\\). The correction joins the smallest terms of the
reconstruction.

## 3. Forming \\(p\\)

- **`atanh`:** \\(1 - t = d_h + d_l\\) exactly (`two_sum`; \\(d_l\\) is
  nonzero for \\(t < 0.5\\)), \\(q_h = 2t \cdot (1/d_h)\\), and one
  correction step,
  \\[
  q_l = \bigl((2t - q_h d_h) - q_h d_l\bigr) \cdot \frac{1}{d_h}
  \\]
  where \\(2t - q_h d_h\\) is exact (`residual`: a fused multiply-add, or
  Dekker's product on the scalar backend). The halving at the end is exact.
- **`asinh`:** \\(t^2 = s_h + s_l\\) exactly (`residual`),
  \\(1 + t^2 = w_h + w_l\\) exactly (`two_sum`), and its square root
  \\(r_h + r_l\\) with one Newton step, as in [`hypot`](./hypot.md):
  \\(r_l = \bigl((w_h - r_h^2) + w_l\bigr)/(2 r_h)\\), \\(r_h^2\\) exact.
  Then \\(r_h - 1\\) and \\(t + (r_h - 1)\\) as `two_sum`s.
- **`acosh`:** \\(t = x - 1\\) is exact for \\(x < 2^{p}\\). \\(t^2 + 2t\\)
  exactly as above, its square root \\(r_h + r_l\\), then \\(t + r_h\\) as a
  `two_sum`. At \\(x = 1\\) the square root is 0 and the Newton step would
  divide by 0, so `acosh(1) = +0` is selected.

## 4. Cut-offs

- **Large inputs:** above \\(2^{20}\\) (`f32`) and \\(2^{40}\\) (`f64`),
  `asinh` and `acosh` are \\(\ln(2t) \pm 1/(4t^2)\\), the correction below
  \\(2^{-22}\\) and \\(2^{-34}\\) ULP. \\(\ln(2t)\\) goes through the same
  path as \\(\ln(1 + p) + 2 \ln 2\\) with \\(p = t/2 - 1\\), exact as a
  `two_sum`: no second logarithm, and \\(t^2\\) never overflows.
- **Small inputs:** below \\(2^{-12}\\) (`f32`) and \\(2^{-28}\\) (`f64`),
  \\(\operatorname{asinh} x = x(1 - x^2/6 + \dots)\\) and
  \\(\operatorname{atanh} x = x(1 + x^2/3 + \dots)\\) round to \\(x\\),
  which is returned. The kernels' inputs are capped there: for smaller
  \\(t\\), the series' \\(s^4\\) would turn subnormal in `f32`, and
  subnormal arithmetic made [`softplus`](./softplus.md) 2.3× slower on x86.

## 5. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Function | Inputs                                                                 | Fused (SIMD) | Unfused (scalar) |
|---------|----------|------------------------------------------------------------------------|--------------|------------------|
| `f32`   | `asinh`  | every finite input                                                     | 0.841        | 0.846            |
| `f32`   | `acosh`  | every finite input from 1                                              | 0.841        | 0.823            |
| `f32`   | `atanh`  | every input in \\((-1, 1)\\)                                           | 0.822        | 0.853            |
| `f64`   | `asinh`  | 24 million samples: \\([-3, 3]\\), \\([0.2, 0.6]\\), and \\(\lvert x\rvert\\) from \\(2^{-30}\\) to \\(2^{60}\\) by magnitude | 0.833 | 0.858 |
| `f64`   | `acosh`  | 24 million samples: \\([1, 4]\\), \\([1, 1.2]\\), and \\(1 + 2^{-50}\\) to \\(2^{60}\\) by magnitude | 0.832 | 0.851 |
| `f64`   | `atanh`  | 32 million samples: \\((-1, 1)\\), \\([0.1, 0.6]\\), and \\(\lvert x\rvert\\) from \\(2^{-30}\\) to 1 by magnitude | 0.826 | 0.845 |

The maxima sit where \\(1 + p\\) crosses \\(\sqrt2\\) and the series
switches between \\(p_h\\) and \\(m - 1\\): \\(\lvert x\rvert \approx 0.35\\)
(`asinh`), \\(x \approx 1.06\\) (`acosh`), \\(\lvert x\rvert \approx 0.17\\)
(`atanh`), as `log1p`'s sit at the ends of its \\(n = 0\\) range. The
`f64` references, `asinh_dd`, `acosh_dd` and `atanh_dd`, form \\(p\\) in
double-double (square roots with a Newton step) and take
[`log1p`](./log1p.md)'s double-double reference of it; above \\(2^{40}\\),
\\(\ln(2t) \pm 1/(4t^2)\\) from [`ln`](./ln.md)'s.

## 6. Cost

`log1p`'s reduction and series with its division, plus, for `atanh`, one
division for \\(1/d_h\\); for `asinh` and `acosh`, a square root and a
division for its Newton step. On AVX2 at 4096 elements, compared with
`std`'s:

| Function | Inputs              | `f32` | `f64` |
|----------|---------------------|-------|-------|
| `asinh`  | \\([-80, 80]\\)     | 2.4×  | 2.5×  |
| `acosh`  | \\([1, 81]\\)       | 2.6×  | 2.7×  |
| `atanh`  | \\([-0.99, 0.99]\\) | 3.3×  | 1.8×  |

Near 0, where `std`'s `f64` `atanh` takes a shorter path, `atanh` is 1.2×
faster (see [Benchmarks](../benchmarks/README.md)).
