# Logistic function `sigmoid`

\\[
\sigma : \mathbb{R} \to [0, 1], \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
\\]

simdly builds `sigmoid` on [`exp`](./exp.md)'s kernel, with the same code
for `f32` and `f64`. Implementation:
[`src/math/sigmoid.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/sigmoid.rs).
Error: **0.782 ULP** (`f32`) and **0.801 ULP** (`f64`) at most, on every
backend, subnormal results included ([measurements](#4-measured-error)).

## 1. Special values and range

| Input                                          | Output      |
|------------------------------------------------|-------------|
| `±0`                                           | `0.5`       |
| `+∞`                                           | `1`         |
| `−∞`                                           | `+0`        |
| NaN                                            | NaN         |
| \\(x\\) above 17.33 (`f32`), 37.43 (`f64`)     | `1`         |
| \\(x\\) below −103.97 (`f32`), −745.13 (`f64`) | `+0`        |
| between −103.97 and −87.34 (`f32`)             | subnormal   |

## 2. A formula that never overflows

With \\(e = e^{-\lvert x\rvert} \in (0, 1]\\):

\\[
\sigma(x) = \begin{cases}
\dfrac{1}{1 + e} & x \ge 0 \\\\[2ex]
\dfrac{e}{1 + e} & x < 0
\end{cases}
\\]

one division per lane, its numerator selected. The textbook
\\(1/(1 + e^{-x})\\) overflows \\(e^{-x}\\) from \\(x \approx -88.7\\)
(`f32`) and returns `0`, although \\(\sigma(x) \approx e^x\\) is
representable down to −103.97.

## 3. Why the formula is not enough as it stands

Rounded naively, that formula reaches **2.4 ULP** in `f64`, measured. Three
errors add up:

- `exp`'s own error, up to 0.84 ULP of \\(e\\), passes into \\(\sigma\\).
  For \\(x < 0\\), \\(\sigma = e/(1 + e)\\) is slightly below \\(e\\); when
  \\(e\\) is just above a power of 2, \\(\sigma\\) falls into the binade
  below, whose ULP is half as large, so the same error counts twice.
- \\(1 + e\\) is rounded, with a ULP twice that of \\(\sigma \in [0.5, 1)\\).
- The division rounds once more.

So the kernel carries a few more bits through each step:

1. **`e` as two floats.** `exp`'s kernel computes \\(e^r = 1 + t\\), with
   `t` a sum it rounds. `exp_parts` keeps that rounding error, and the one
   of \\(1 + t\\), both exact (`two_sum`), as a low part:
   \\(e = (y_h + y_l) \cdot 2^k\\). `exp` itself returns \\(y_h \cdot 2^k\\),
   as before.
2. **\\(1 + e\\) exactly**, as \\(d_h + d_l\\): since \\(e \le 1\\),
   \\(d_l = (e_h - (d_h - 1)) + e_l\\) is exact (Fast2Sum).
3. **One correction step** after the division. With \\(n\\) the numerator
   (\\(y_h + y_l\\), unscaled, or 1) and \\(q = n_h \cdot (1/d_h)\\):
   \\[
   \sigma \cdot 2^{-s} = q + \frac{(n_h - q\,d_h) + n_l - q\,d_l}{d}
   \\]
   where \\(n_h - q\,d_h\\) is exact (`residual`: a fused multiply-add, or
   Dekker's product on the scalar backend). The correction is tiny, so the
   reciprocal \\(1/d_h\\) used for it needs no more precision.
4. **Scale last**: \\(\cdot 2^k\\) for \\(x < 0\\) in two normal factors, as
   in `exp`, so subnormal results round in one step at the end.

Two cut-offs keep every intermediate in range. \\(-\lvert x\rvert\\) is
capped at −112 (`f32`) and −753 (`f64`), where \\(\sigma\\) already rounds to
`+0`. And for \\(1 + e\\), \\(k\\) is capped at −40 (`f32`) and −100 (`f64`):
\\(e\\) below that is about \\(2^{-15}\\) ULP of the result, and without the
cap \\(y_l \cdot 2^k\\) would turn subnormal, which costs over 4× in
throughput on x86.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                          | Fused (SIMD) | Unfused (scalar) |
|---------|-----------------------------------------------------------------|--------------|------------------|
| `f32`   | every finite input                                              | 0.782        | 0.782            |
| `f64`   | 32 million samples over \\([-753, 40]\\), double-double reference | 0.801        | 0.801            |

Both maxima are subnormal results, rounded once in the scaled domain and
once scaling back. Normal `f64` results stay at or below 0.69 ULP. The `f64`
reference, `sigmoid_dd`, divides in double-double arithmetic using
[`exp`](./exp.md)'s double-double reference.

## 5. Cost

Per register: `exp`'s reduction and polynomial, one division, about ten
additions and multiply-adds for the extra bits and the correction, four
`ldexp`, and selects. On AVX2 at 4096 elements, `sigmoid` is 2.0× faster
than the scalar \\(1/(1 + e^{-x})\\) for `f32` and 1.6× for `f64`, which are
less accurate than this (see [Benchmarks](../benchmarks/README.md)).
