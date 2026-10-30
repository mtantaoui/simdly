# Natural logarithm `ln`

\\[
\ln : (0, \infty) \to \mathbb{R}, \qquad e^{\ln x} = x
\\]

simdly uses fdlibm's algorithm: musl's `log.c` for `f64`, and the `logf.c`
musl shipped before its table-driven version for `f32`, which stays in
`f32` arithmetic. Implementation:
[`src/math/ln.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/ln.rs).
Error: **0.835 ULP** (`f32`) and **0.823 ULP** (`f64`) at most, on every
backend, subnormal inputs included ([measurements](#6-measured-error)).

## 1. Special values

| Input     | Output   |
|-----------|----------|
| `1`       | `+0`, exactly |
| `±0`      | `−∞`     |
| `x < 0`, `−∞` | NaN  |
| `+∞`      | `+∞`     |
| NaN       | NaN      |

## 2. Range reduction

\\[
x = m \cdot 2^n, \qquad m \in [\tfrac{\sqrt2}{2}, \sqrt2)
\\]

so \\(\ln x = n \ln 2 + \ln m\\). [`frexp`](./cbrt.md#6-frexp-and-ldexp)
gives \\(m \in [1, 2)\\); above \\(\sqrt2\\), \\(m\\) is halved and \\(n\\)
increased, both exactly. Centring \\(m\\) on 1 keeps
\\(f = m - 1 \in [-0.29, 0.41)\\) small, and \\(f\\) is exact
(Sterbenz). The threshold is musl's, bit for bit: `0x3fb504f3` for `f32`,
and for `f64` only the high 32 bits of \\(\sqrt2\\), as musl compares the
high word.

**Subnormal inputs** are first scaled by \\(2^{24}\\) (`f32`) or
\\(2^{54}\\) (`f64`), and \\(n\\) lowered to match.

## 3. The approximation

With \\(s = \frac{f}{2 + f}\\), so that \\(1 + f = \frac{1 + s}{1 - s}\\):

\\[
\ln(1 + f) = \ln(1 + s) - \ln(1 - s) = 2s + \frac{2s^3}{3} + \frac{2s^5}{5} + \cdots = 2s + s \cdot R(s^2)
\\]

an odd series in \\(s\\), with \\(\lvert s\rvert < 0.172\\). fdlibm fits
\\(R\\) with 4 coefficients for `f32` (error \\(2^{-34.2}\\)) and 7 for
`f64`, split by the parity of the powers of \\(w = s^4\\) into two
independent Horner chains, as in [`atan`](./atan.md#3-the-polynomial). One
division per register, for \\(s\\).

## 4. Reconstruction

\\(2s\\) itself is not used: with \\(h = f^2/2\\), the identity
\\(2s = f - s f\\) gives

\\[
\ln(1 + f) = f - h + s\,(h + R)
\\]

where \\(f\\) is exact, \\(h\\) is close to the part of \\(f\\) that the
series takes back, and \\(s\,(h + R)\\) is a small correction. Adding
\\(n \ln 2\\) as `hi + lo`, musl's order is

\\[
\ln x = \Bigl(\bigl(s\,(h + R) + n \cdot \mathtt{lo}\bigr) - h + f\Bigr) + n \cdot \mathtt{hi}
\\]

small terms first, the exact \\(n \cdot \mathtt{hi}\\) (trailing zero bits
in `hi`) last. On the SIMD backends, \\(s\,(h + R) + n \cdot \mathtt{lo}\\)
is one fused multiply-add.

This is what makes \\(\ln\\) accurate **next to 1**, where the result is
tiny and \\(f\\) carries it almost alone. Evaluating
\\(2s + s \cdot R\\) directly instead reaches 1.13 ULP at
\\(x = 1.0000006\\) in `f32`.

## 5. Why each piece is there

Measured by removing it, on the regular test inputs (`f32`):

| Removed                                  | Error                 |
|------------------------------------------|-----------------------|
| the \\(\sqrt2\\) adjustment (\\(m \in [1, 2)\\)) | 1.05 ULP       |
| \\(\ln 2\\)'s `lo` part                  | 160 ULP               |
| the subnormal scaling                    | 745 000 ULP           |
| the \\(h\\) trick (\\(2s + s \cdot R\\)) | 1.13 ULP, next to 1   |

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                         | Fused (SIMD) | Unfused (scalar) | Subnormal inputs |
|---------|----------------------------------------------------------------|--------------|------------------|------------------|
| `f32`   | every positive finite input                                    | 0.835        | 0.835            | 0.503            |
| `f64`   | 330 000 samples: every exponent, around 1 and \\(\sqrt2\\); 200-bit reference | 0.823 | 0.823  | 0.488            |

The worst cases sit next to the \\(\sqrt2\\) threshold, where \\(f\\) and
\\(s\\) are largest. The `f64` tests use `std`'s `ln` corrected in
double-double arithmetic as the reference (see
[Accuracy](../accuracy.md)), and still pass with `std`'s `ln` deliberately
off by 3 ULP.

## 7. Cost

Per register: `frexp` and two selects, one division, a 4- or 7-term
series, and the reconstruction. On AVX2, `ln` is 3× faster than `std` for
`f32` and 1.8× for `f64`: glibc's `log` is table-driven and already fast
(see [Benchmarks](../benchmarks/README.md)).
