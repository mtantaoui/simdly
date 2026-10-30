# Tangent `tan`

\\[
\tan : \mathbb{R} \setminus \left(\tfrac{\pi}{2} + \pi\mathbb{Z}\right) \to \mathbb{R}, \qquad \tan x = \frac{\sin x}{\cos x}
\\]

in radians, for every finite input (no float is a pole). The reduction is
[`sin`'s](./sin.md#2-range-reduction), the kernels musl's `k_tanf.c` and
`k_tan.c`. Implementation:
[`src/math/trig.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/trig.rs).
Error: **0.797 ULP** (`f32`) and **0.802 ULP** (`f64`) at most, on every
backend ([measurements](#4-measured-error)).

## 1. Special values

| Input      | Output |
|------------|--------|
| `±0`       | `±0`   |
| `±∞`, NaN  | NaN    |

## 2. Quadrants

With \\(x = n \cdot \frac{\pi}{2} + y\\), \\(\lvert y\rvert \le \frac{\pi}{4}\\):
\\(\tan x = \tan y\\) for even \\(n\\), \\(-1/\tan y\\) for odd \\(n\\). Both
zeros (even multiples of \\(\frac{\pi}{2}\\)) and poles (odd ones) cancel
in the reduction: without `sin`'s round 3, \\(\tan(413441.447\ldots)\\)
measures 2.17 ULP.

## 3. Kernels

- **`f32`**, in `f64`: a degree-13 odd polynomial (musl's `k_tanf`),
  then \\(-1/t\\) for odd lanes, rounded once to `f32`. The polynomial's
  own error, \\(2^{-25.5}\\), is why `f32` reaches 0.797 ULP rather than
  `sin`'s 0.501.
- **`f64`** (musl's `k_tan`): a degree-27 polynomial in \\(y_0 + y_1\\).
  Above \\(\lvert y\rvert = 0.6744\\) it is evaluated at
  \\(\frac{\pi}{4} - \lvert y\rvert\\) and \\(\tan y\\) recovered from
  \\(\tan\left(\frac{\pi}{4} - a\right) = \frac{1 - \tan a}{1 + \tan a}\\).
  For odd lanes, \\(-1/w\\) rounded alone is up to 2 ULP off; one
  correction fixes it: \\(a = -1/w\\), then \\(a + a(1 + a w + a e)\\), with
  \\(1 + a w\\) exact (`residual`) and \\(e\\) the rounding error of
  \\(w = x + r\\).

Each lane needs at most one division, for whichever form it uses; a
register whose lanes need neither (\\(\lvert y\rvert < 0.6744\\), even
\\(n\\)) skips it.

## 4. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against
`sin` / `cos` of [`sin`'s reference](./sin.md#6-measured-error), divided in
double-double:

| Element | Inputs                                                          | Max   |
|---------|-----------------------------------------------------------------|-------|
| `f32`   | 3·10⁵ samples of every binade to `f32::MAX`                     | 0.797 |
| `f64`   | 3·10⁵ samples of every binade to `f64::MAX`                     | 0.802 |
| `f64`   | the 2 910 inputs below \\(2^{20} \cdot \frac{\pi}{2}\\) within \\(2^{-62}\\) of a multiple of \\(\frac{\pi}{2}\\) (one-off run) | 0.500 |
| `f64`   | 2 000 neighbours of the closest input in each binade of \\([2^{20}, 2^{33})\\) (one-off run) | 0.505 |

An ignored test checks every `f32` against 1 ULP
(`cargo test --release -- --ignored tan_f32_exhaustive`).

## 5. Cost

The reduction, one polynomial, at most one division. On AVX2, `tan` is
about 4–8× faster than `std` for `f32` and 2–2.2× for `f64` (see
[Benchmarks](../benchmarks/README.md)); by argument range, `f32` 1.8–4.7×
and `f64` 1.2–4.3×.
