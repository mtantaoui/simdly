# Hyperbolic functions `sinh` and `cosh`

\\[
\sinh x = \frac{e^x - e^{-x}}{2}, \qquad \cosh x = \frac{e^x + e^{-x}}{2}
\\]

simdly builds both on [`exp`](./exp.md)'s kernel, with the same code for
`f32` and `f64`. Implementation:
[`src/math/hyperbolic.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/hyperbolic.rs).
Error: **0.711 ULP** (`f32`) and **0.728 ULP** (`f64`) at most, for both
functions, on every backend ([measurements](#4-measured-error)).

## 1. Special values and range

| Input                                        | `sinh`       | `cosh`  |
|----------------------------------------------|--------------|---------|
| `±0`                                         | `±0`         | `1`     |
| `±∞`                                         | `±∞`         | `+∞`    |
| NaN                                          | NaN          | NaN     |
| tiny \\(x\\)                                 | \\(x\\), subnormals included | `1` |
| \\(\lvert x\rvert\\) above 89.42 (`f32`), 710.48 (`f64`) | `±∞` | `+∞` |

The results stay finite up to \\(\ln(2 \cdot \mathrm{MAX})\\), past
\\(e^x\\)'s own overflow at 88.72 (`f32`) and 709.78 (`f64`).

## 2. One exponential, both terms

With \\(a = \lvert x\rvert\\), `exp_parts` (see [`sigmoid`](./sigmoid.md))
gives \\(E = e^a = (y_h + y_l) \cdot 2^k\\) to a few bits more than the
element type. Its reciprocal needs no second exponential:

\\[
\frac{1}{E} = (q_h + q_l) \cdot 2^{-k}, \qquad q_h = \frac{1}{y_h}, \qquad
q_l = \bigl((1 - q_h y_h) - q_h y_l\bigr)\, q_h
\\]

where \\(1 - q_h y_h\\) is exact (`residual`: a fused multiply-add, or
Dekker's product on the scalar backend). Then

\\[
\cosh x,\ \sinh a = 2^{k-1} \Bigl((y_h + y_l) \pm (q_h + q_l)\,2^{-2k}\Bigr)
\\]

with the high parts summed exactly (`two_sum`), so \\(E \pm 1/E\\) rounds
once, and the factor \\(2^{k-1}\\) applied last in two normal steps, as in
`exp`. That last step is what keeps results finite between 88.72 and 89.42
(`f32`).

\\(2^{-2k}\\) is capped at \\(2^{-60}\\) (`f32`) and \\(2^{-120}\\)
(`f64`), where \\(e^{-a}\\) no longer shows in the result: without the cap,
\\(q_l \cdot 2^{-2k}\\) would turn subnormal, which is many times slower on
x86.

## 3. `sinh` near 0

\\(E - 1/E\\) cancels as \\(a \to 0\\): the bits the subtraction loses
expose the error of `exp`'s polynomial, which for `f32` is only about
\\(2^{-27.7}\\). Below \\(\lvert x\rvert = 0.5\\), `sinh` uses its Taylor
series instead:

\\[
\sinh x = x + x \cdot z \cdot P(z), \qquad z = x^2, \qquad
P(z) = \frac{1}{3!} + \frac{z}{5!} + \frac{z^2}{7!} + \dots
\\]

to \\(x^9\\) (`f32`) and \\(x^{15}\\) (`f64`), truncated below
\\(2^{-35}\\) and \\(2^{-64}\\) of the result. \\(x\\) is exact and the rest
is below 5% of the result, so the sum rounds about once. `cosh` has no
cancellation and keeps the exponential form everywhere.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Function       | Inputs                                                       | Fused (SIMD) | Unfused (scalar) |
|---------|----------------|--------------------------------------------------------------|--------------|------------------|
| `f32`   | `sinh`, `cosh` | every finite input                                           | 0.708        | 0.711            |
| `f64`   | `sinh`, `cosh` | 32 million samples over \\([-712, 712]\\), \\([-3, 3]\\), \\([-0.6, 0.6]\\), double-double reference | 0.728 | 0.728 |

The maxima sit where \\(e^{-a}\\) no longer matters, \\(x \approx 27.4\\)
(`f32`) and \\(261\\) (`f64`): the error there is `exp_parts`'s own,
rounded once. The `f64` references, `sinh_dd` and `cosh_dd`, use
[`exp`](./exp.md)'s double-double reference, and below \\(\lvert x\rvert =
1\\) `sinh_dd` sums its Taylor series in double-double.

## 5. Cost

`exp`'s reduction and polynomial, one more division (\\(1/y_h\\)), a few
additions and multiply-adds for the low parts, two `ldexp` for
\\(2^{-2k}\\) and two for the final scaling; `sinh` adds its series and a
select. On AVX2 at 4096 elements in \\([-80, 80]\\), `sinh` is 3.3× faster
than `std`'s for `f32` and 2.2× for `f64`, and `cosh` 4.0× and 1.9× (see
[Benchmarks](../benchmarks/README.md)).
