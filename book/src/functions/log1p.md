# Logarithm of one plus `log1p`

\\[
\operatorname{log1p} : (-1, \infty) \to \mathbb{R}, \qquad \operatorname{log1p}(x) = \ln(1 + x)
\\]

`(1.0 + x).ln()` loses everything \\(1 + x\\) rounds away: for
\\(\lvert x\rvert\\) below half an ULP of 1, it returns `0` instead of
\\(\approx x\\), and for small \\(x\\) it keeps only the bits of \\(x\\) that
survive the addition. simdly follows fdlibm's `s_log1p.c` on top of
[`ln`](./ln.md)'s reduction and series, with the same code for `f32` and
`f64`. Implementation:
[`src/math/log1p.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/log1p.rs).
Error: **0.835 ULP** (`f32`) and **0.844 ULP** (`f64`) at most, on every
backend ([measurements](#4-measured-error)).

## 1. Special values

| Input           | Output           |
|-----------------|------------------|
| `±0`            | `±0`             |
| tiny \\(x\\)    | \\(x\\), subnormals included |
| `−1`            | `−∞`             |
| `x < −1`, `−∞`  | NaN              |
| `+∞`            | `+∞`             |
| NaN             | NaN              |

## 2. `1 + x` without losing `x`

`two_sum` forms \\(1 + x = u + c\\) exactly: \\(u\\) is the rounded sum and
\\(c\\) what it dropped. For \\(x > -1\\), \\(u\\) is positive and normal,
so it reduces as in `ln`:

\\[
u = m \cdot 2^n, \qquad m \in [\tfrac{\sqrt2}{2}, \sqrt2)
\\]

and

\\[
\ln(1 + x) = n \ln 2 + \ln(1 + f) + \ln\Bigl(1 + \frac{c}{u}\Bigr)
\approx n \ln 2 + \ln(1 + f) + \frac{c}{u}
\\]

with \\(f = m - 1\\), exact. The last step drops
\\((c/u)^2/2 < 2^{-2p-1}\\), far below an ULP.

**When \\(n = 0\\)**, which is \\(x \in [\tfrac{\sqrt2}{2} - 1, \sqrt2 - 1)\\),
the range where \\(\ln(1 + x)\\) is small, \\(f = u - 1\\) would lose \\(c\\)
again. But there \\(1 + f = 1 + x\\) holds exactly with \\(f = x\\), so the
series takes \\(x\\) itself and needs no correction.

## 3. The series and reconstruction

Exactly `ln`'s: \\(s = f/(2 + f)\\),
\\(\ln(1 + f) = f - \mathrm{hfsq} + s\,(\mathrm{hfsq} + R(s^2))\\), summed
small terms first. \\(c/u\\) joins \\(n \cdot \ln2_{lo}\\) among the
smallest terms:

\\[
\Bigl(\bigl(s\,(\mathrm{hfsq} + R) + (n \cdot \ln2_{lo} + c/u)\bigr) - \mathrm{hfsq} + f\Bigr) + n \cdot \ln2_{hi}
\\]

For \\(x\\) far above \\(2^p\\), \\(u = x\\) and \\(c = 1\\): \\(c/u\\) is
negligible and `log1p` becomes `ln`, as it should.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                    | Fused (SIMD) | Unfused (scalar) |
|---------|-----------------------------------------------------------|--------------|------------------|
| `f32`   | every input above −1                                      | 0.835        | 0.835            |
| `f64`   | 32 million samples: \\((-1, 10^6)\\), and \\(\pm 10^{-60}\\) to \\(10^{300}\\) by magnitude, double-double reference | 0.844 | 0.839 |

The maxima sit near the ends of the \\(n = 0\\) range, as `ln`'s sit near
1. The `f64` reference, `log1p_dd`, sums the Taylor series in double-double
below \\(\lvert x\rvert = 2^{-6}\\), and above uses [`ln`](./ln.md)'s
double-double reference on \\(u\\) plus \\(c/u\\).

## 5. Cost

`ln`'s, plus `two_sum` (six additions), a select and one division for
\\(c/u\\). On AVX2 at 4096 elements, `log1p` costs 6–8% more than `ln`, and
is 4.4× faster than `std`'s `ln_1p` for `f32` and 3.2× for `f64` (see
[Benchmarks](../benchmarks/README.md)).
