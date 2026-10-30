# Cosine `cos`

\\[
\cos : \mathbb{R} \to [-1, 1]
\\]

in radians, for every finite input. `cos` is [`sin`](./sin.md) with the
quadrant shifted by one, \\(\cos x = \sin\left(x + \frac{\pi}{2}\right)\\):
the same reduction, polynomials and huge-argument path, one code path
for both, specialised at compile time. Implementation:
[`src/math/trig.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/trig.rs).
Error: **0.501 ULP** (`f32`) and **0.730 ULP** (`f64`) at most, on every
backend ([measurements](#3-measured-error)).

## 1. Special values

| Input      | Output |
|------------|--------|
| `±0`       | `1`    |
| `±∞`, NaN  | NaN    |

## 2. Quadrants

With \\(x = n \cdot \frac{\pi}{2} + y\\), \\(\lvert y\rvert \le \frac{\pi}{4}\\)
([`sin`, Sec. 2](./sin.md#2-range-reduction)):

| \\(n \bmod 4\\) | 0 | 1 | 2 | 3 |
|-----------------|---|---|---|---|
| \\(\cos x\\)    | \\(\cos y\\) | \\(-\sin y\\) | \\(-\cos y\\) | \\(\sin y\\) |

`cos` cancels near the **odd** multiples of \\(\frac{\pi}{2}\\), `sin` near
the even ones. The `f64` round 3 is needed for both: without it,
\\(\cos(413441.447\ldots)\\) measures 1.06 ULP.

## 3. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against the
same reference as `sin`, shifted by one quadrant:

| Element | Inputs                                                          | Max   |
|---------|-----------------------------------------------------------------|-------|
| `f32`   | 3·10⁵ samples of every binade to `f32::MAX`                     | 0.501 |
| `f64`   | 3·10⁵ samples of every binade to `f64::MAX`                     | 0.730 |
| `f64`   | the 2 910 inputs below \\(2^{20} \cdot \frac{\pi}{2}\\) within \\(2^{-62}\\) of a multiple of \\(\frac{\pi}{2}\\) (one-off run) | 0.500 |
| `f64`   | 2 000 neighbours of the closest input in each binade of \\([2^{20}, 2^{33})\\) (one-off run) | 0.500 |

An ignored test checks every `f32` against 1 ULP
(`cargo test --release -- --ignored cos_f32_exhaustive`).

## 4. Cost

The same as `sin`, minus its sign-of-zero select, with the same speed-ups
by argument range ([`sin`, Sec. 7](./sin.md#7-cost)): on AVX2, `cos` is
about 3× faster than `std` for `f32` and 2.7× for `f64` (see
[Benchmarks](../benchmarks/README.md)).
