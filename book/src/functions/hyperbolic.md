# Hyperbolic functions `sinh`, `cosh` and `tanh`

\\[
\sinh x = \frac{e^x - e^{-x}}{2}, \qquad \cosh x = \frac{e^x + e^{-x}}{2},
\qquad \tanh x = \frac{\sinh x}{\cosh x}
\\]

simdly builds all three on [`exp`](./exp.md)'s kernel, with the same code
for `f32` and `f64`. Implementation:
[`src/math/hyperbolic.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/hyperbolic.rs).
Error at most, on every backend ([measurements](#5-measured-error)):

| Function       | `f32`         | `f64`         |
|----------------|---------------|---------------|
| `sinh`, `cosh` | **0.711 ULP** | **0.728 ULP** |
| `tanh`         | **0.684 ULP** | **0.689 ULP** |

## 1. Special values and range

| Input                                        | `sinh`       | `cosh`  | `tanh` |
|----------------------------------------------|--------------|---------|--------|
| `±0`                                         | `±0`         | `1`     | `±0`   |
| `±∞`                                         | `±∞`         | `+∞`    | `±1`   |
| NaN                                          | NaN          | NaN     | NaN    |
| tiny \\(x\\)                                 | \\(x\\), subnormals included | `1` | \\(x\\), subnormals included |
| \\(\lvert x\rvert\\) above 9.01 (`f32`), 19.06 (`f64`) | | | `±1` |
| \\(\lvert x\rvert\\) above 89.42 (`f32`), 710.48 (`f64`) | `±∞` | `+∞` | `±1` |

`sinh` and `cosh` stay finite up to \\(\ln(2 \cdot \mathrm{MAX})\\),
past \\(e^x\\)'s own overflow at 88.72 (`f32`) and 709.78 (`f64`).

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

## 4. `tanh`

\\[
\tanh a = \frac{N}{D}, \qquad D = (y_h + y_l) + (q_h + q_l)\,2^{-2k}
\\]

both scaled by \\(2^{1-k}\\), which cancels. \\(D\\) is \\(E + 1/E\\)
summed exactly as for `cosh`, \\(d_h + d_l\\). From \\(a = 0.5\\) on,
\\(N\\) is \\(E - 1/E\\) the same way. Below, where that cancels, \\(N\\)
is `sinh`'s series kept as two floats, \\(n_h = a\\) and
\\(n_l = a \cdot z \cdot P(z)\\), each times \\(2^{1-k}\\), exact as
\\(k \in \{0, 1\}\\) there.

The quotient takes one division and one correction step, as in
[`sigmoid`](./sigmoid.md):

\\[
q = n_h \cdot \frac{1}{d_h}, \qquad
\tanh a = q + \frac{(n_h - q\,d_h) + n_l - q\,d_l}{d_h}
\\]

with \\(n_h - q\,d_h\\) exact (`residual`). Below \\(2^{-12}\\) (`f32`)
and \\(2^{-28}\\) (`f64`), \\(\tanh x = x(1 - x^2/3 + \dots)\\) rounds to
\\(x\\), which is returned; the kernel's input is capped there, so
\\(a \cdot z\\) stays normal. Far from 0, \\(N\\) and \\(D\\) agree to
every bit kept and the quotient rounds to 1: no overflow, as \\(E\\) is
never scaled by \\(2^k\\).

## 5. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Function       | Inputs                                                       | Fused (SIMD) | Unfused (scalar) |
|---------|----------------|--------------------------------------------------------------|--------------|------------------|
| `f32`   | `sinh`, `cosh` | every finite input                                           | 0.708        | 0.711            |
| `f64`   | `sinh`, `cosh` | 32 million samples over \\([-712, 712]\\), \\([-3, 3]\\), \\([-0.6, 0.6]\\), double-double reference | 0.728 | 0.728 |
| `f32`   | `tanh`         | every finite input                                           | 0.662        | 0.684            |
| `f64`   | `tanh`         | 32 million samples over \\([-25, 25]\\), \\([-1, 1]\\), and \\(\lvert x\rvert\\) from \\(2^{-30}\\) to \\(2^5\\) by magnitude, double-double reference | 0.660 | 0.689 |

The maxima sit where \\(e^{-a}\\) no longer matters, \\(x \approx 27.4\\)
(`f32`) and \\(261\\) (`f64`): the error there is `exp_parts`'s own,
rounded once. The `f64` references, `sinh_dd` and `cosh_dd`, use
[`exp`](./exp.md)'s double-double reference, and below \\(\lvert x\rvert =
1\\) `sinh_dd` sums its Taylor series in double-double. `tanh`'s maxima
sit just below 0.5, at the series' end; its reference, `tanh_dd`, divides
`sinh_dd` by `cosh_dd` in double-double.

## 6. Cost

`exp`'s reduction and polynomial, one more division (\\(1/y_h\\)), a few
additions and multiply-adds for the low parts, two `ldexp` for
\\(2^{-2k}\\) and two for the final scaling; `sinh` adds its series and a
select. `tanh` takes both, plus one division and its correction. On AVX2
at 4096 elements, compared with `std`'s:

| Function | Inputs           | `f32` | `f64` |
|----------|------------------|-------|-------|
| `sinh`   | \\([-80, 80]\\)  | 3.3×  | 2.2×  |
| `cosh`   | \\([-80, 80]\\)  | 4.0×  | 1.9×  |
| `tanh`   | \\([-5, 5]\\)    | 2.3×  | 2.0×  |
| `tanh`   | \\([-80, 80]\\)  | 1.0×  | 0.9×  |

Over \\([-80, 80]\\) most inputs give \\(\pm1\\), which `std`'s
`tanh` returns early; SIMD lanes compute every input in full, so there
`tanh` only matches `std`'s (`f32`) or is 10% slower (`f64`). The
benchmark uses \\([-5, 5]\\), as activations mostly are (see
[Benchmarks](../benchmarks/README.md)).
