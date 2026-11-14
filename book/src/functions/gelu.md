# Activation `gelu`

\\[
\operatorname{gelu}(x) = x\,\Phi(x) = \frac{x}{2}\Bigl(1 + \operatorname{erf}\frac{x}{\sqrt2}\Bigr)
\\]

The Gaussian error linear unit, \\(\Phi\\) the standard normal CDF, in one
pass instead of the five of its composite form (burn's default: divide,
`erf`, add, multiply, divide). Implementation:
[`src/math/gelu.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/gelu.rs).
Error: **0.500 ULP** (`f32`, every input) and **0.838 ULP** (`f64`) at
most, on every backend, negative inputs included
([measurements](#4-measured-error)).

## 1. Special values

| Input  | Result |
|--------|--------|
| `±0`   | `±0`   |
| `+∞`   | `+∞`   |
| `−∞`   | `−0`, the limit (the composite form gives NaN: \\(-\infty \cdot 0\\)) |
| NaN    | NaN    |

From \\(x \approx -14.4\\) (`f32`) or \\(-38.6\\) (`f64`) down, the result
rounds to `−0`.

## 2. Why not \\(\tfrac{x}{2}(1 + \operatorname{erf})\\)

For \\(x < 0\\), \\(\operatorname{erf}(x/\sqrt2)\\) tends to \\(-1\\) and
\\(1 + \operatorname{erf}\\) cancels: at \\(x = -5\\) it is about
\\(5.7 \cdot 10^{-7}\\), ten `f32` ULPs of 1. In `f32` the composite form
returns \\(-1.490 \cdot 10^{-6}\\) there (the result is
\\(-1.433 \cdot 10^{-6}\\): 4% off), \\(-1.64 \cdot 10^{-7}\\) at \\(-5.5\\)
(\\(-1.04 \cdot 10^{-7}\\)), and from \\(x \approx -5.55\\) on, `−0`. The rounding of \\(x/\sqrt2\\)
alone costs \\(2a^2\\) times its own relative error in
\\(\operatorname{erfc} a\\): about 25 ULPs at \\(a = 5\\) (\\(x \approx -7\\)).

## 3. The kernel

fdlibm's rational functions for `erf` (see [`erf`](./erf.md)), in
\\(a = \lvert x\rvert/\sqrt2\\), regrouped in \\(x\\) so that \\(x/\sqrt2\\)
is never rounded:

- \\(\lvert x\rvert < 0.84375\sqrt2\\): \\(\operatorname{erf} a = a(1 + r)\\),
  \\(r = P/Q\\) in \\(z = a^2 = x^2/2\\), exact, so
  \\(\operatorname{gelu} x = x/2 + x^2(1 + r)/(2\sqrt2)\\).
- up to \\(1.25\sqrt2\\): \\(\operatorname{erf} a = \mathit{erx} + R(s)\\),
  \\(s = a - 1\\) computed to twice the precision, so
  \\(\operatorname{gelu}(-\lvert x\rvert) = -\lvert x\rvert\bigl((1 - \mathit{erx})/2 - R/2\bigr)\\).
- beyond: \\(\operatorname{erfc} a = e^{-a^2 - 0.5625 + R/S}/a\\), \\(R/S\\)
  in \\(1/a^2 = 2/x^2\\); the \\(1/a\\) cancels against \\(x/2\\):
  \\[
  \operatorname{gelu}(-\lvert x\rvert) = -e^{-x^2/2 - 0.5625 - \ln\sqrt2 + R/S}
  \\]
  with \\(x^2\\) exact and the exponent in two parts, one
  `exp_parts` (see [`sigmoid`](./sigmoid.md)), scaled last so that
  subnormal results round once.

For \\(x > 0\\), \\(\operatorname{gelu} x = x + \operatorname{gelu}(-x)\\),
where the second term is small. For \\(x < 0\\), though, the result is
much smaller than the terms of \\(P/Q\\), \\(R\\) and \\(R/S\\): evaluated
as they stand, rounding at each Horner step and in the division, they cost
up to 2.9 ULPs. So for `f64`:

- the last two Horner steps (four for \\(R/S\\)) carry their rounding
  errors and the argument's low part along (compensated Horner); the
  earlier steps' errors reach the result scaled down by powers of the
  argument;
- the quotient gets one correction step, with \\(p - q_h r_h\\) exact;
- everything after is in two parts, rounded once at the end.

`f32` runs in `f64` (two registers of `f64` per register of `f32`), where
the plain evaluation is far more precise than `f32` needs. In `f32`
itself, fdlibm's coefficients rounded to `f32` would already cost more
than half an ULP.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against a
double-double reference, `gelu_dd`, written in \\(x\\) itself:
\\(\Phi(x) = \tfrac12 + \varphi(x) \sum x^{2n+1}/(2n+1)!!\\) for
\\(\lvert x\rvert \le 6\\), the Mills ratio's continued fraction beyond.

| Element | Inputs                                                                      | Every backend |
|---------|-----------------------------------------------------------------------------|---------------|
| `f32`   | every input (`cargo test --release -- --ignored gelu_f32_exhaustive`)       | 0.500         |
| `f64`   | 140 million: 20 million in each of seven ranges, the cuts and the underflow | 0.838         |

The `f64` maxima are at \(x \approx -1.14\), where the first range's
cancellation is largest, and \(-1.19\) and \(-1.82\), the ends of the
middle range; positive inputs stay below 0.57.

## 5. Cost

On AVX2 at 4096 elements, compared with the composite form on `libm`'s
`erf`, element by element:

| Inputs           | `f32` | `f64` |
|------------------|-------|-------|
| \\(x \in [-5, 5]\\) | 1.55× | 1.31× |

Three ranges, each with its rational function, and `exp` for the third:
with inputs spread over all three, every register computes all of them,
as for `erf`. The compensated steps cost `f64` about 45% more than the
plain evaluation. See [Benchmarks](../benchmarks/README.md).
