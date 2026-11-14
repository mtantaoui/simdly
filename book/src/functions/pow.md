# Powers `powf` and `powi`

\\[
\operatorname{pow}(x, y) = x^y = e^{\,y \ln x}
\\]

Three entry points share one kernel, with the same code for `f32` and
`f64`: `powf` (an exponent per element), `powf_scalar` (one exponent for
the slice) and `powi` (an `i32` exponent). Implementation:
[`src/math/pow.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/pow.rs).
Error: **0.804 ULP** (`f32`) and **0.797 ULP** (`f64`) at most, on every
backend, overflow and subnormal results included
([measurements](#5-measured-error)).

## 1. Special values

C99's, as `f64::powf`:

| Case                                         | Result                    |
|----------------------------------------------|---------------------------|
| `y = ±0`, any `x` (NaN included)             | `1`                       |
| `x = 1`, any `y` (NaN included)              | `1`                       |
| `x = −1`, `y = ±∞`                           | `1`                       |
| `x < 0` finite, `y` not an integer           | NaN                       |
| `x < 0`, `y` an integer                      | \\(\lvert x\rvert^y\\), negated for odd `y` |
| `x = ±0`, `y < 0`                            | `+∞`, or `−∞` for `x = −0` and odd `y` |
| `x = ±0`, `y > 0`                            | `+0`, or `−0` for `x = −0` and odd `y` |
| `y = ±∞`                                     | `+∞` or `+0` by whether \\(\lvert x\rvert\\) is above or below 1 |
| `x = ±∞`                                     | as \\(\lvert x\rvert = \infty\\), the sign of `x` for odd `y` |
| other NaN                                    | NaN                       |

## 2. Why `pow` is hard

An absolute error \\(\delta\\) in \\(y \ln x\\) becomes a relative error
\\(\delta\\) in \\(e^{y \ln x}\\). For finite `f64` results,
\\(\lvert y \ln x\rvert\\) reaches 745, about \\(2^{9.5}\\): to keep the
result within a fraction of an ULP, \\(\ln x\\) must be good to about
\\(2^{-66}\\) relative, twelve bits beyond `f64`. For `f32`, with
\\(\lvert y \ln x\rvert \le 104\\), about \\(2^{-35}\\). Composing a
correctly rounded `ln` with `exp` cannot get there.

## 3. The kernel

1. **\\(\ln \lvert x\rvert = l_h + l_l\\), to about twice the precision.**
   \\(\lvert x\rvert = m \cdot 2^n\\) as in [`ln`](./ln.md),
   \\(f = m - 1\\) exact, and \\(s = f/(2 + f)\\) as two floats:
   \\(2 + f\\) is exact as a Fast2Sum, and the quotient gets one correction
   step with the exact residual \\(f - s_h(2 + f)\\). Then
   \\[
   \ln(1 + f) = 2 \operatorname{atanh} s
   = 2s\Bigl(1 + \frac{z}{3} + \frac{z^2}{5} + z^3 W(z)\Bigr), \qquad z = s^2,\ \lvert s\rvert \le 0.1716
   \\]
   with \\(W\\) the Taylor series to \\(z^{12}/25\\) (`f64`) or
   \\(z^7/15\\) (`f32`), evaluated in the element type (Estrin's scheme for
   `f64`, for a shorter dependency chain); \\(s\\), \\(z\\), \\(1/3\\),
   \\(1/5\\) and every product above \\(W\\) are two-part. Finally
   \\(n \ln 2\\), with \\(\ln 2\\) in two parts, the first with trailing
   zeros so that \\(n \cdot \ln2_{hi}\\) is exact, and the sum renormalised
   so that \\(l_l\\) is below half an ULP of \\(l_h\\).
2. **\\(y \ln \lvert x\rvert = p_h + p_l\\)**, the product exact
   (`residual`: a fused multiply-add, or Dekker's product on the scalar
   backend). `y` may itself come in two parts (below).
3. **\\(e^{p_h + p_l} = (e_h + e_l)(1 + p_l) \cdot 2^k\\)** from
   `exp_parts` (see [`sigmoid`](./sigmoid.md)), scaled last in two normal
   factors as in [`exp`](./exp.md), so that results overflow, underflow and
   turn subnormal by themselves, rounding once.

There are no tables: musl's `pow` looks \\(\ln\\) up in a 128-entry table,
which SIMD lanes would need gathers for, and NEON has none. The special
cases of § 1 are selected at the end, and only when some lane of the
register has \\(x \le 0\\) or a non-finite `x` or `y`.

## 4. `powi`

`f32::powi` multiplies repeatedly, rounding at every step, and loses about
\\(\log_2 \lvert n\rvert\\) ULPs. simdly's `powi`:

- \\(n = 0, 1, 2, -1\\): `1`, `x`, `x·x`, `1/x`, each rounded once.
- \\(3 \le \lvert n\rvert \le 8\\): binary powering in two-part arithmetic,
  every product exact to about twice the precision, rounded once at the
  end; for \\(n < 0\\), the reciprocal with one correction step. Lanes
  where \\(\lvert x^{\lvert n\rvert}\rvert\\) leaves
  \\([2^{e_{min} + p}, 2^{e_{max} - 28}]\\) (where low parts would lose
  bits, or the scalar backend's Dekker product would overflow), and
  \\(0\\), \\(\pm\infty\\), NaN, take the kernel above.
- Otherwise the kernel, with \\(y = n\\). `f32` cannot hold every `i32`,
  so \\(n = n_h + n_l\\), both exact (\\(\lvert n_l\rvert \le 64\\)), and
  \\(y \ln \lvert x\rvert = (n_h + n_l)(l_h + l_l)\\) in two parts. With
  \\(x = 1 + 2^{-23}\\) and \\(n = 2^{24} + 1\\), for instance, rounding
  \\(n\\) to `f32` would cost two ULPs.

## 5. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against a
double-double reference, `pow_dd` (\\(y \ln x\\) from [`ln`](./ln.md)'s
double-double reference, then [`exp`](./exp.md)'s):

| Element | Inputs                                                                         | Fused (SIMD) | Unfused (scalar) |
|---------|--------------------------------------------------------------------------------|--------------|------------------|
| `f32`   | 30 million pairs: `x` over every binade with \\(y \ln x\\) across the whole range, `x` near 1 with large `y`, and \\(x \in (0, 10]\\), \\(y \in [-10, 10]\\) | 0.804 | 0.804 |
| `f64`   | the same, 30 million pairs                                                       | 0.797        | 0.797            |
| `f32`   | `powi`, \\(\lvert n\rvert \le 8\\), every binade, both signs                     | 0.778        | 0.778            |
| `f64`   | `powi`, \\(\lvert n\rvert \le 8\\), every binade, both signs                     | 0.761        | 0.761            |
| `f32`   | `powi`, `n` up to \\(\pm 2^{31}\\), `x` near 1                                  | 0.765        | 0.765            |
| `f64`   | `powi`, `n` up to \\(\pm 2^{31}\\)                                              | 0.682        | 0.682            |

The `powi` maxima for \\(\lvert n\rvert \le 8\\) are lanes that fall back to
the kernel; binary powering itself rounds once.

## 6. Cost

\\(\ln \lvert x\rvert\\) in two parts dominates: about three times `ln`'s
work, two divisions in all with `exp_parts`'. On AVX2 at 4096 elements,
compared with `std`:

| Function             | Inputs                                     | `f32` | `f64` |
|----------------------|--------------------------------------------|-------|-------|
| `powf`               | \\(x \in [0.01, 10]\\), \\(y \in [-6, 6]\\) | 1.34× | 1.18× |
| `powf_scalar(2.5)`   | \\(x \in [0.01, 10]\\)                      | 1.32× | 1.17× |
| `powi(3)`            | \\(x \in [0.01, 10]\\)                      | 0.21× | 0.25× |
| `powi(7)`            | \\(x \in [0.01, 10]\\)                      | 0.22× | 0.20× |

`std`'s `powi` is two or three multiplications per element, rounded at
each; simdly's spends about four to five times that on two-part arithmetic
to round once. Exponents beyond \\(\pm 8\\) take the kernel, about as fast
as `powf`. See [Benchmarks](../benchmarks/README.md).
