# Arc sine `asin`

\\[
\operatorname{asin} : [-1, 1] \to [-\tfrac{\pi}{2}, \tfrac{\pi}{2}], \qquad \sin(\operatorname{asin} x) = x
\\]

`asin` shares its ingredients with [`acos`](./acos.md): the rational
approximation \\(r(z)\\), π/2 as `hi + lo`, and the exact split of
\\(\sqrt{z}\\). Read that page first; this one covers what differs.
Implementation:
[`src/math/asin.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/asin.rs).
Error: **< 1 ULP** for `f32` and `f64`, on every backend
([measurements](#6-measured-error)).

## 1. Special values

| Input            | Output    | How                                            |
|------------------|-----------|------------------------------------------------|
| `±0`             | `±0`      | \\(x + x \cdot r(0) = x\\), exactly            |
| `±1`             | `±π/2`    | set explicitly ([Sec. 4](#4-the-endpoints)); nearest float |
| `\|x\| > 1`, `±∞`  | NaN       | square root of a negative number               |
| NaN              | NaN       | propagated                                     |

## 2. Two ranges

\\(\operatorname{asin}\\) is odd, so the large range works on
\\(a = \lvert x\rvert\\) and restores the sign at the end (negation is exact).

| Range                         | Formula                                                              | Result in                                    |
|-------------------------------|----------------------------------------------------------------------|----------------------------------------------|
| \\(\lvert x\rvert < 0.5\\)    | \\(x + x \cdot r(x^2)\\)                                             | \\((-\frac{\pi}{6}, \frac{\pi}{6})\\)        |
| \\(\lvert x\rvert \ge 0.5\\)  | \\(\frac{\pi}{2} - 2\operatorname{asin}(s)\\), \\(s = \sqrt{z}\\), \\(z = \frac{1 - a}{2}\\) | \\(\pm[\frac{\pi}{6}, \frac{\pi}{2}]\\) |

The second line is \\(\operatorname{asin}(a) = \frac{\pi}{2} - \operatorname{acos}(a)\\)
with the half-angle form of [`acos`](./acos.md#3-three-ranges-one-approximation).
As there, \\(z\\) is exact and \\(\operatorname{asin}\\) is only ever
approximated on \\([0, \frac12]\\).

**\\(\lvert x\rvert < 0.5\\)** needs nothing more: the result is close to
\\(x\\), the correction \\(x \cdot r(x^2)\\) is at most 4.7 % of it, and on
the SIMD backends \\(x \cdot r + x\\) is one fused multiply-add, a single
rounding.

## 3. \\(\lvert x\rvert \ge 0.5\\): where the precision goes

\\[
\operatorname{asin}(a) = \tfrac{\pi}{2} - 2\bigl(s + s \cdot r(z)\bigr)
\\]

Near \\(a = 0.5\\) this cancels: \\(s = 0.5\\), so the result
\\(\frac{\pi}{2} - 1.05 \approx 0.52\\) is a third of \\(\frac{\pi}{2}\\).
The half-ULP rounding of \\(s\\), doubled and multiplied by the slope of
\\(2\operatorname{asin}\\) (up to 2.31), is worth 0.58 ULP of the result on
its own. So, as in `acos`, \\(s\\) is split into \\(d_f\\) (its high 12
bits in `f32`, 21 in `f64`) and the correction
\\(c = \frac{z - d_f^2}{s + d_f} \approx \sqrt{z} - d_f\\).

That leaves the order of the operations. Three were measured, over every
`f32` in \\([0.5, 1)\\) and 280 000 `f64` samples against a 200-bit
reference (max ULP, fused / unfused):

| Formula for \\(\operatorname{asin}(a)\\)                                                                   | Source              | `f32`         | `f64`         |
|------------------------------------------------------------------------------------------------------------|---------------------|---------------|---------------|
| A: \\(\frac{\mathtt{hi}}{2} - \bigl(2s\,r - (\mathtt{lo} - 2c) - (\frac{\mathtt{hi}}{2} - 2d_f)\bigr)\\)   | musl `asin.c`       | 0.967 / 0.981 | 0.848 / 0.865 |
| B: \\(\mathtt{hi} - \bigl(2(s + s\,r) - \mathtt{lo}\bigr)\\), no split                                      | musl, for \\(a > 0.975\\) | 2.431 / 2.442 | 2.119 / 2.151 |
| **C: \\((\mathtt{hi} - 2d_f) - \bigl(2(s\,r + c) - \mathtt{lo}\bigr)\\)**                                  | simdmath            | **0.758 / 0.767** | **0.660 / 0.679** |

musl's `asin.c` uses A up to \\(a = 0.975\\) and B above it, where \\(s\\)
is small and B is accurate (0.63 ULP). Its `asinf.c` avoids the question by
computing this range in `f64`, which SIMD cannot do cheaply. **simdly uses
C over the whole range**: the most accurate in both types, and one formula,
so no extra select.

C works because its first term is **exact**: \\(d_f\\) has so few bits
that \\(\mathtt{hi} - 2d_f\\) lands on a float. This was checked on every
`f32` in the range and 2 million `f64` samples. Every rounding error then
sits in the small second term, \\(2(s\,r + c) - \mathtt{lo}\\), and only
the final subtraction rounds at the scale of the result.

## 4. The endpoints

At \\(a = 1\\), \\(z = s = d_f = 0\\) and \\(c = 0/0\\) is NaN, so
\\(\operatorname{asin}(\pm 1)\\) is set to \\(\pm(\mathtt{hi} + \mathtt{lo})\\),
which rounds to the float nearest π/2. For \\(a > 1\\), \\(z < 0\\) and the
square root gives NaN.

## 5. On SIMD

Per register, as for `acos`: one mask `small` = \\(\lvert x\rvert < 0.5\\),
**one** \\(z\\) selected per lane (\\(x^2\\) or \\(\frac{1 - a}{2}\\)), so
one \\(r(z)\\) and one \\(\sqrt{z}\\); then the two results, the
\\(a = 1\\) case and the sign are merged by selects. The cost is again two
divisions and one square root per register.

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Backend           | Inputs                        | \\(\lvert x\rvert < 0.5\\) | \\(\lvert x\rvert \ge 0.5\\) |
|---------|-------------------|-------------------------------|----------------------------|------------------------------|
| `f32`   | SIMD (fused)      | all \\(2^{32}\\)              | 0.708                      | 0.758                        |
| `f32`   | scalar (unfused)  | all \\(2^{32}\\)              | 0.743                      | 0.767                        |
| `f64`   | SIMD (fused)      | ~2.5 × 10⁵, 200-bit reference | 0.593                      | 0.660                        |
| `f64`   | scalar (unfused)  | ~2.5 × 10⁵, 200-bit reference | 0.636                      | 0.679                        |

## 7. Speed

See [Benchmarks](../benchmarks/README.md).
