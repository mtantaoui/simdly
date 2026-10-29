# Arc cosine `acos`

\\[
\operatorname{acos} : [-1, 1] \to [0, \pi], \qquad \cos(\operatorname{acos} x) = x
\\]

simdly uses musl's algorithm (`acosf.c`, `acos.c`, from Sun's fdlibm),
written once for every backend in
[`src/math/acos.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/acos.rs).
Error: **< 1 ULP** for `f32` and `f64`, on every backend
([measurements](#9-measured-error)).
[`asin`](./asin.md) reuses the same pieces.

## 1. Special values

| Input            | Output    | How                                         |
|------------------|-----------|---------------------------------------------|
| `1`              | `+0`      | set explicitly ([Sec. 7](#7-the-endpoints)) |
| `-1`             | `π`       | nearest float, from the `x ≤ −0.5` case     |
| `±0`             | `π/2`     | nearest float, from the `\|x\| < 0.5` case    |
| `\|x\| > 1`, `±∞`  | NaN       | square root of a negative number            |
| NaN              | NaN       | propagated                                  |

## 2. Why not just `π/2 − asin(x)`?

The identity holds everywhere, but near \\(x = 1\\) it computes a small
result as the difference of two numbers close to \\(\pi/2\\):

\\[
\operatorname{acos}(1 - \delta) \approx \sqrt{2\delta}
\qquad\text{while}\qquad
\operatorname{asin}(1 - \delta) \approx \tfrac{\pi}{2} - \sqrt{2\delta}
\\]

The rounding error of \\(\operatorname{asin}\\) is about one ULP *of π/2*,
which is many ULPs of the small result: for \\(x = 1 - 2^{-20}\\) in `f32`,
\\(\operatorname{acos}(x) \approx 0.0014\\), whose ULP is 1000 times smaller
than that of \\(\pi/2\\). The fix is to never form the result by
cancellation.

## 3. Three ranges, one approximation

The half-angle formula \\(\cos\theta = 1 - 2\sin^2(\theta/2)\\), with
\\(\theta = \operatorname{acos} x\\), gives
\\(\sin(\theta/2) = \sqrt{(1 - x)/2}\\), so

\\[
\operatorname{acos}(x) = 2 \operatorname{asin}\sqrt{\tfrac{1 - x}{2}}
\qquad
\operatorname{acos}(x) = \pi - 2 \operatorname{asin}\sqrt{\tfrac{1 + x}{2}}
\\]

the second one from \\(\operatorname{acos}(x) = \pi - \operatorname{acos}(-x)\\).
Each range picks the identity that keeps the argument of
\\(\operatorname{asin}\\) small:

| Range              | Identity                                              | `asin` at                              | Result in                    |
|--------------------|-------------------------------------------------------|----------------------------------------|------------------------------|
| \\(\lvert x\rvert < 0.5\\)    | \\(\frac{\pi}{2} - \operatorname{asin}(x)\\)          | \\(t = x\\)                            | \\((\frac{\pi}{3}, \frac{2\pi}{3})\\) |
| \\(x \le -0.5\\)   | \\(\pi - 2\operatorname{asin}(s)\\)                   | \\(s = \sqrt{z},\ z = \frac{1 + x}{2}\\) | \\([\frac{2\pi}{3}, \pi]\\)  |
| \\(x \ge 0.5\\)    | \\(2\operatorname{asin}(s)\\)                         | \\(s = \sqrt{z},\ z = \frac{1 - x}{2}\\) | \\([0, \frac{\pi}{3}]\\)     |

In every range \\(\operatorname{asin}\\) is needed only on
\\([-\frac12, \frac12]\\), and \\(z = t^2 \le \frac14\\).

**\\(z\\) is exact.** For \\(x \in [0.5, 1]\\), \\(1 - x\\) is exact
(Sterbenz's lemma: \\(a - b\\) is exact when \\(b/2 \le a \le 2b\\)), and
halving is exact. The same holds for \\(1 + x\\) on \\([-1, -0.5]\\). So the
reduction itself adds no error; only \\(\sqrt{z}\\) rounds.

## 4. Approximating `asin` on \\([0, \frac12]\\)

\\(\operatorname{asin}\\) is odd, with Taylor series

\\[
\operatorname{asin}(t) = t + \frac{t^3}{6} + \frac{3t^5}{40} + \frac{15t^7}{336} + \cdots
= t + t \cdot R(t^2), \qquad R(z) = \frac{z}{6} + \frac{3z^2}{40} + \cdots
\\]

The series converges slowly towards \\(t = \frac12\\), so musl approximates
\\(R\\) with a rational function, its coefficients fitted to minimise the
maximum error on \\([0, \frac14]\\):

\\[
R(z) \approx r(z) = \frac{z \, p(z)}{q(z)}
\\]

| Element | \\(p\\)    | \\(q\\)    | Coefficients | Divisions |
|---------|------------|------------|--------------|-----------|
| `f32`   | degree 2   | degree 1   | 4            | 1         |
| `f64`   | degree 5   | degree 4   | 10           | 1         |

The constant term of \\(p\\) is close to \\(\frac16\\) (`f32`: 0.16666587,
`f64`: 0.16666666666666666), matching the series. Both polynomials are
evaluated with Horner's rule, one multiply-add per coefficient.

**Why the approximation error hardly matters.** The correction is small
next to \\(t\\): at most
\\(R(\frac14) = \frac{\operatorname{asin}(1/2)}{1/2} - 1 = \frac{\pi}{3} - 1 \approx 0.047\\).
A relative error \\(\epsilon\\) in \\(r\\) therefore moves the result by at
most \\(0.047\,\epsilon\\) relative: the error budget goes to the final
roundings instead.

## 5. π/2 in two parts

π/2 is not a float. It is stored as two:

\\[
\tfrac{\pi}{2} = \mathtt{hi} + \mathtt{lo}
\\]

| Element | `hi` (π/2 rounded down)       | `lo`                     | `lo` in ULPs of π/2 |
|---------|-------------------------------|--------------------------|---------------------|
| `f32`   | `0x3fc90fda` = 1.5707962513   | 7.5497894 × 10⁻⁸         | 0.63                |
| `f64`   | `0x3ff921fb54442d18`          | 6.1232340 × 10⁻¹⁷        | 0.28                |

Using `hi` alone would cost up to 0.63 ULP in `f32`, most of the budget.
Instead, `lo` is folded into the small terms *before* they meet `hi`, so
only the last operation rounds at the scale of the result.

**\\(|x| < 0.5\\):**

\\[
\operatorname{acos}(x) = \mathtt{hi} - \bigl(x - (\mathtt{lo} - x \cdot r(x^2))\bigr)
\\]

**\\(x \le -0.5\\):**

\\[
\operatorname{acos}(x) = 2\Bigl(\mathtt{hi} - \bigl(s + (s \cdot r(z) - \mathtt{lo})\bigr)\Bigr)
\\]

Multiplying by 2 is exact. Here \\(s\\) is used as rounded, and that is
fine: its error is at most half an ULP of \\(s < \frac12\\), the slope of
\\(2\operatorname{asin}(s)\\) is at most \\(2/\sqrt{3/4} \approx 2.31\\), and
the result in \\([2, \pi]\\) has an ULP 8 times that of \\(s\\): at most
\\(0.5 \times 2.31 / 8 \approx 0.15\\) ULP.

## 6. \\(x \ge 0.5\\): recovering \\(\sqrt{z}\\) exactly

Here \\(\operatorname{acos}(x) \approx 2s\\): the result has the *same
relative precision* as \\(s\\), so the half-ULP rounding of
\\(s = \sqrt{z}\\) would pass straight into the result. musl recovers it.

1. **Split.** \\(d_f\\) is \\(s\\) with its low bits cleared: 12 significant
   bits kept in `f32`, 21 in `f64`. Then \\(d_f^2\\) has at most 24 or 42
   bits and is **exact**.
2. **Exact remainder.** \\(d_f\\) agrees with \\(\sqrt{z}\\) to 11 bits,
   so \\(d_f^2\\) is within a factor 2 of \\(z\\), and \\(z - d_f^2\\) is
   exact by Sterbenz's lemma.
3. **Correction.** Since \\(z - d_f^2 = (\sqrt{z} - d_f)(\sqrt{z} + d_f)\\),

\\[
c = \frac{z - d_f^2}{s + d_f} \approx \sqrt{z} - d_f
\\]

   to almost full relative precision: \\(c\\) is small, so its own rounding
   is tiny in absolute terms.

\\(d_f + c\\) now represents \\(\sqrt{z}\\) to about twice the working
precision, and

\\[
\operatorname{acos}(x) = 2\bigl(d_f + (s \cdot r(z) + c)\bigr)
\\]

adds the small terms first and the exact \\(d_f\\) last. (\\(s \cdot r(z)\\)
may use the rounded \\(s\\): it is a correction below 5 % of the result.)

## 7. The endpoints

- **\\(x = -1\\):** \\(z = 0\\), \\(s = 0\\), \\(r(0) = 0\\), so the
  \\(x \le -0.5\\) formula gives
  \\(2(\mathtt{hi} - (0 - \mathtt{lo})) = 2(\mathtt{hi} + \mathtt{lo})\\):
  \\(\mathtt{hi} + \mathtt{lo}\\) rounds to the float nearest π/2, and
  doubling it gives the float nearest π, exactly `f32::consts::PI` or
  `f64::consts::PI`.
- **\\(x = 1\\):** \\(z = s = d_f = 0\\), so \\(c = 0/0\\) is NaN. The result
  is set to \\(0\\) explicitly.
- **\\(|x| > 1\\), \\(\pm\infty\\):** \\(z < 0\\), and \\(\sqrt{z}\\) is
  NaN, which propagates.

## 8. On SIMD: every lane, every case

Lanes of one register can fall in different ranges, so there are no
branches. Per register:

1. Two masks: `small` = \\(|x| < 0.5\\), `neg` = \\(x < 0\\).
2. **One** \\(z\\), selected per lane: \\(x^2\\), \\(\frac{1 + x}{2}\\) or
   \\(\frac{1 - x}{2}\\). So \\(r(z)\\) and \\(\sqrt{z}\\) are computed once,
   not three times.
3. The three results of Sec. 5–6, then two selects, and one more for
   \\(x = 1\\).

The cost is dominated by two divisions (in \\(r\\) and \\(c\\)) and one
square root per register; the rest are multiply-adds, compares and
selects. Lanes with \\(|x| < 0.5\\) compute a square root they do
not use: cheaper than a branch that would split the register.

On the SIMD backends, \\(\mathtt{lo} - x \cdot r\\),
\\(s \cdot r - \mathtt{lo}\\), \\(s \cdot r + c\\) and every Horner step are
fused multiply-adds, one rounding instead of two. The scalar backend rounds
twice, exactly as musl does.

## 9. Measured error

Maximum error in ULPs, per range ([how it is measured](../accuracy.md)):

| Element | Backend               | Inputs                         | \\(\lvert x\rvert < 0.5\\) | \\(x \le -0.5\\) | \\(x \ge 0.5\\) |
|---------|-----------------------|--------------------------------|-----------------|------------------|-----------------|
| `f32`   | SIMD (fused)          | all \\(2^{32}\\)               | 0.796           | 0.914            | 0.740           |
| `f32`   | scalar (unfused)      | all \\(2^{32}\\)               | 0.804           | 0.918            | 0.754           |
| `f64`   | SIMD (fused)          | 4.2 × 10⁵, 200-bit reference   | 0.766           | 0.879            | 0.617           |
| `f64`   | scalar (unfused)      | 4.2 × 10⁵, 200-bit reference   | 0.774           | 0.891            | 0.625           |

The largest errors are near \\(x = \pm 0.5\\), where \\(z\\) approaches
\\(\frac14\\) and the correction \\(t \cdot r(z)\\) is largest.

## 10. Speed

See [Benchmarks](../benchmarks/README.md).
