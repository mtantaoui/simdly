# `softplus` and `log_sigmoid`

\\[
\operatorname{softplus}(x) = \ln(1 + e^x), \qquad
\operatorname{log\_sigmoid}(x) = \ln \sigma(x) = -\operatorname{softplus}(-x)
\\]

`softplus` is a smooth \\(\max(x, 0)\\); `log_sigmoid` is the logarithm of
the [logistic function](./sigmoid.md), as in a binary cross-entropy. Both
share one kernel, with the same code for `f32` and `f64`. Implementation:
[`src/math/softplus.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/softplus.rs).
Error: **0.867 ULP** (`f32`) and **0.838 ULP** (`f64`) at most, on every
backend, subnormal results included ([measurements](#4-measured-error)).
`log_sigmoid(x)` is `-softplus(-x)` bit for bit, so its errors are the same.

## 1. Special values and range

| Input                                          | `softplus`  | `log_sigmoid` |
|------------------------------------------------|-------------|---------------|
| `±0`                                           | \\(\ln 2\\) | \\(-\ln 2\\)  |
| `+∞`                                           | `+∞`        | `−0`          |
| `−∞`                                           | `+0`        | `−∞`          |
| NaN                                            | NaN         | NaN           |
| \\(x\\) above 20 (`f32`), 40 (`f64`)           | \\(x\\)     | \\(\approx -e^{-x}\\) |
| \\(x\\) below −20 (`f32`), −40 (`f64`)         | \\(\approx e^x\\) | \\(x\\) |

`softplus(x) ≈ eˣ` far below 0 is subnormal from −87.34 (`f32`) and
−708.4 (`f64`) and rounds to `+0` past `exp`'s underflow threshold;
`log_sigmoid` mirrors it far above 0.

## 2. Why not the formula as it stands

`(1 + x.exp()).ln()` overflows `eˣ` from \\(x \approx 88.7\\) (`f32`) and
returns `∞`, although the result is about \\(x\\); and for \\(x\\) below about
−17 (`f32`), \\(1 + e^x\\) rounds to 1 and the result to 0 instead of
\\(e^x\\). The usual fix,

\\[
\operatorname{softplus}(x) = \max(x, 0) + \operatorname{log1p}(e^{-\lvert x\rvert})
\\]

never overflows, but rounds the result up to three times: `exp`'s error
(up to 0.91 ULP in `f32`) passes into `log1p`'s argument, `log1p` rounds
again, and for \\(x > 0\\) so does the addition. Built from simdly's own
`exp` and `log1p`, it reaches **1.62 ULP** in `f32`, measured over every
input.

## 3. One logarithm, of a number formed exactly

`exp_parts` (see [`sigmoid`](./sigmoid.md)) gives
\\(e^x = (y_h + y_l) \cdot 2^k\\) to a few bits more than the element type.
With \\(K = \max(k, 0)\\):

\\[
1 + e^x = 2^K w, \qquad w = (y_h + y_l)\,2^{k - K} + 2^{-K} \in [0.7, 2.5]
\\]

so \\(\operatorname{softplus}(x) = K \ln 2 + \ln w\\), one logarithm with
no separate rounding of \\(x\\). `two_sum` forms \\(w = w_h + w_l\\)
exactly, and \\(\ln w\\) proceeds as in [`ln`](./ln.md): \\(w_h = m \cdot
2^n\\), \\(m \in [\tfrac{\sqrt2}{2}, \sqrt2)\\), \\(f = m - 1\\) exact,

\\[
\ln(w_h + w_l) = (n + K) \ln 2 + \ln(1 + f) + \frac{w_l}{w_h}
\\]

with \\(w_l / w_h\\) among the smallest terms of the reconstruction, as in
[`log1p`](./log1p.md). Also as in `log1p`: for \\(x < 0\\) and \\(n = 0\\),
\\(1 + e^x\\) may round, and \\(f = w_h - 1\\) would then cancel against
\\(w_l\\) (1.14 ULP, measured). There \\(1 + f = 1 + e_h\\) exactly with
\\(f = e_h\\), the high part of \\(e^x\\), so the series takes \\(e_h\\) itself
and only \\(e_l / w_h\\) is added.

Two cut-offs:

- Above 20 (`f32`) and 40 (`f64`), the result is \\(x\\): \\(e^{-x}\\) is
  below half an ULP. The input to `exp_parts` is capped there, which keeps
  \\(2^{-K}\\) normal.
- For \\(k\\) below −29 (`f32`) and −100 (`f64`), \\(\ln(1 + e) = e(1 - e/2
  + \dots)\\) is \\(e\\) to 0.02 ULP or better, and the result is
  \\((y_h + y_l) \cdot 2^k\\), scaled last as in `exp` so subnormal results
  round once. The `f32` cut-off is also where the series' \\(s^4\\) would
  turn subnormal, which costs over 2× in throughput on x86.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                                       | Fused (SIMD) | Unfused (scalar) |
|---------|------------------------------------------------------------------------------|--------------|------------------|
| `f32`   | every finite input                                                           | 0.857        | 0.867            |
| `f64`   | 32 million samples over \\([-760, 45]\\), \\([-40, 5]\\) and \\([-1, 1]\\), double-double reference | 0.836 | 0.838 |

The maxima sit near \\(x \approx -0.9\\), where \\(w\\) crosses \\(\sqrt2\\).
The `f64` reference, `softplus_dd`, is \\(\max(x, 0)\\) plus the logarithm
of \\(1 + e^{-\lvert x\rvert}\\) in double-double, from
[`exp`](./exp.md)'s and [`log1p`](./log1p.md)'s references; below 0 it
keeps the result scaled by \\(e^x\\)'s \\(2^k\\), as subnormal results
need.

## 5. Cost

`exp`'s reduction and polynomial, then `ln`'s reduction and series, joined
by `two_sum` and three `ldexp` (two more to scale tiny results); three
divisions in all. They form one long
dependency chain, so `softplus` costs about 1.7× `exp` and `log1p` taken
separately. On AVX2 at 4096 elements, `softplus` is 1.9× faster than the
scalar \\(\max(x, 0) + \operatorname{log1p}(e^{-\lvert x\rvert})\\) for
`f32` and 1.2× for `f64`, which are less accurate than this (see
[Benchmarks](../benchmarks/README.md)).
