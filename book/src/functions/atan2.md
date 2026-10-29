# Two-argument arc tangent `atan2`

\\[
\operatorname{atan2}(y, x) \in [-\pi, \pi], \text{ the angle of the point } (x, y)
\\]

`atan2` builds on [`atan`](./atan.md): same ranges, constants and
polynomial. musl's approach, \\(\operatorname{atan}(\lvert y/x\rvert)\\)
followed by a quadrant fix-up, reaches 1.47 ULP; simdly removes its two
extra roundings to stay **< 1 ULP**. Implementation:
[`src/math/atan2.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/atan2.rs).

## 1. Special values

As C99 Annex F (and Rust's `f32::atan2`):

| `y`        | `x`             | Result      |
|------------|-----------------|-------------|
| `±0`       | `+0` or `> 0`   | `±0`        |
| `±0`       | `−0` or `< 0`   | `±π`        |
| `≠ 0`      | `±0`            | `±π/2`      |
| finite     | `+∞`            | `±0`        |
| finite     | `−∞`            | `±π`        |
| `±∞`       | finite          | `±π/2`      |
| `±∞`       | `+∞`            | `±π/4`      |
| `±∞`       | `−∞`            | `±3π/4`     |
| NaN        | any             | NaN         |
| any        | NaN             | NaN         |

The sign of the result is the sign bit of `y`, and the half-plane is chosen
by the sign bit of `x`: `x = −0` counts as negative.

## 2. Where musl loses precision

With \\(Y = \lvert y\rvert\\), \\(X = \lvert x\rvert\\), musl computes
\\(z = \operatorname{atan}(Y/X)\\), then \\(z\\) or \\(\pi - z\\), and the
sign. Two roundings happen outside `atan`'s careful evaluation:

1. **The quotient \\(Y/X\\).** For small ratios, \\(\operatorname{atan}(a) \approx a\\):
   the quotient *is* the answer, and its half-ULP rounding passes straight
   through. Worse, when \\(a\\) is just above a power of 2 and
   \\(\operatorname{atan}(a)\\) just below it, the result's ULP is half that
   of \\(a\\): the quotient's rounding alone is worth up to **1 ULP**.
2. **\\(\pi - z\\).** \\(z\\) is already rounded, then rounded again.

Measured on 30 million random `f32` pairs: **1.47 ULP** at worst.

## 3. No rounded quotient

`atan`'s reduction \\(t = \frac{p\,a - q}{p + q\,a}\\) with \\(a = Y/X\\)
becomes, multiplying through by \\(X\\),

\\[
t = \frac{p\,Y - q\,X}{p\,X + q\,Y}
\\]

with the ranges tested as \\(Y < c \cdot X\\). The numerators stay exact,
as in `atan`. The division still rounds, so its error is recovered:

\\[
\mathit{inv} = \frac{1}{\mathit{den}}, \qquad
t = \mathit{num} \cdot \mathit{inv}, \qquad
t_\ell = (\mathit{num} - t \cdot \mathit{den}) \cdot \mathit{inv}
\\]

\\(\mathit{num} - t \cdot \mathit{den}\\) is **exact** (it is the remainder
of the division), so \\(t + t_\ell\\) represents the quotient to about
twice the working precision. It enters the result to first order:
\\(\operatorname{atan}(t + t_\ell) \approx \operatorname{atan}(t) + \frac{t_\ell}{1 + t^2}\\),
with \\(\frac{1}{1+t^2} \approx 1 - t^2\\) (good enough for a correction
already \\(2^{-24}\\) smaller than \\(t\\)). One division per register, as
in `atan`, plus a few multiply-adds.

The exact remainder is a new `Float` operation, `residual(a, b, c) = c − a·b`:

| Backend | How                                                                                     |
|---------|-----------------------------------------------------------------------------------------|
| SIMD    | a fused multiply-add: one rounding of an exactly representable result, so exact        |
| scalar  | Dekker's product: \\(a \cdot b = p + e\\) exactly, from half-width splits of \\(a\\) and \\(b\\), then \\((c - p) - e\\) |

Without it, the scalar backend (separate multiply and add) gets the
remainder wrong and loses the gain.

## 4. π folded into the constants

For \\(x < 0\\):

\\[
\pi - \operatorname{atan}(c) - \operatorname{atan}(t)
= H - \bigl((\tau \cdot S - L) - \tau\bigr), \qquad \tau = -t
\\]

where \\(H + L\\) is the hi + lo split of \\(\pi - \operatorname{atan}(c)\\)
for each range's \\(c\\) (computed at 300 bits). This is `atan`'s
reconstruction with \\(\tau\\) for \\(t\\): one final rounding, in every
quadrant.

| \\(c\\) | \\(\pi - \operatorname{atan}(c)\\) | `f32` `H`    | `f64` `H`             |
|---------|------------------------------------|--------------|-----------------------|
| 0       | \\(\pi\\)                          | `0x40490fdb` | `0x400921fb54442d18`  |
| 0.5     | \\(2.6779\ldots\\)                 | `0x402b6374` | `0x40056c6e7397f5ae`  |
| 1       | \\(3\pi/4\\)                       | `0x4016cbe4` | `0x4002d97c7f3321d2`  |
| 1.5     | \\(2.1588\ldots\\)                 | `0x400a29c3` | `0x400145385fa3af71`  |
| ∞       | \\(\pi/2\\)                        | `0x3fc90fdb` | `0x3ff921fb54442d18`  |

## 5. Scaling and special values

- **Overflow.** \\(p\,X + q\,Y\\) overflows for inputs near the largest
  float, so both are scaled by \\(2^{-60}\\) (`f32`) or \\(2^{-500}\\)
  (`f64`) when the larger is above \\(2^{60}\\) or \\(2^{500}\\).
- **Underflow.** For a subnormal \\(Y\\), the remainder falls below the
  smallest float. Both are scaled up by the same factor when the smaller is
  below \\(2^{-60}\\) or \\(2^{-500}\\). Powers of 2 keep the ratio exact.
- **Infinities** become 1, and the other argument 0 (or NaN, as
  \\(\mathrm{NaN} \cdot 0\\)): \\((\infty, \infty)\\) is then
  \\(\operatorname{atan2}(1, 1) = \frac{\pi}{4}\\), and so on.
- **\\((\pm 0, \pm 0)\\)** takes \\(X = 1\\): only the signs matter.
- **Signs** use a second new `Float` operation, `is_sign_negative`, so
  that \\(x = -0\\) selects the \\(\pi\\) half-plane.

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)). `f32`: 40
million pairs per variant, every quadrant, ratios over 60 binades,
including ratios near \\(\tan(2^{-k})\\) (where the result crosses a power
of 2), against `f64`'s `atan2`. `f64`: 300 000 pairs against a 200-bit
reference.

| Variant                                     | `f32` max | `f32` pairs > 1 ULP | `f64` max |
|---------------------------------------------|-----------|---------------------|-----------|
| Folded reduction, no remainder              | 1.499     | 79 362              |           |
| + remainder, then musl's \\(\pi - z\\)      | 0.998     | 0                   |           |
| **+ remainder, π folded in (simdly)**       | **0.808** | **0**               | **0.794** |

Fused (SIMD) and scalar (Dekker remainder) give the same maxima. Folding
\\(\pi\\) in matters for the margin, and for the special values: with
\\(\pi - z\\), \\(\operatorname{atan2}(\infty, -\infty)\\) misses the
float nearest \\(\frac{3\pi}{4}\\).

The tests check 200 000 pairs per type on every backend, the special
values above bit for bit, and 10⁸ more `f32` pairs in an ignored test
(`cargo test --release -- --ignored many_pairs`).

## 7. Cost

Per register, on top of `atan`: the scaling and remapping (a few compares
and selects), the remainder (two multiply-adds), and the constants picked
per quadrant (two more selects). simdly's `atan2` is 2.6× faster than
`std` for `f32` and 1.6× for `f64` on AVX2, less than `atan`, the price of
the extra precision. See [Benchmarks](../benchmarks/README.md).
