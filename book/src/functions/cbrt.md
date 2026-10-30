# Cube root `cbrt`

\\[
\operatorname{cbrt} : \mathbb{R} \to \mathbb{R}, \qquad (\sqrt[3]{x})^3 = x
\\]

Defined for negative numbers too: \\(\sqrt[3]{-8} = -2\\). Implementation:
[`src/math/cbrt.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/cbrt.rs).
Error: **0.667 ULP** at most, for `f32` and `f64`, on every backend
([measurements](#7-measured-error)).

## 1. Special values

| Input    | Output   |
|----------|----------|
| `±0`     | `±0`     |
| `±∞`     | `±∞`     |
| NaN      | NaN      |
| `x < 0`  | \\(-\sqrt[3]{\lvert x\rvert}\\) |

Exact cubes (\\(27\\), \\(-8\\), \\(0.125\\)) give exact results.

## 2. Why not musl's `cbrtf`

musl's `cbrtf` gets a 5-bit first guess by dividing the float's **bits by 3
as an integer** (the exponent divides by 3, roughly), then refines it with
two Newton steps in **double precision**. Neither suits SIMD: there is no
vector integer division, and `f32` lanes would have to be widened to `f64`.

simdly instead reduces the argument exactly with two new `Float`
operations, `frexp` and `ldexp` ([Sec. 6](#6-frexp-and-ldexp)), and keeps
musl's final step, from its `f64` `cbrt.c`.

## 3. Exact reduction

With \\(\lvert x\rvert = m \cdot 2^e\\), \\(m \in [1, 2)\\), write
\\(e = 3k + r\\) with \\(r \in \\{-1, 0, 1\\}\\):

\\[
\sqrt[3]{\lvert x\rvert} = \sqrt[3]{m \cdot 2^r} \cdot 2^k
\\]

- \\(k = \operatorname{round}(e/3)\\), computed as
  \\((e \cdot \tfrac13 + R) - R\\) with \\(R = 1.5 \cdot 2^{23}\\) (or
  \\(2^{52}\\)): adding \\(R\\) pushes the fraction bits out, a
  round-to-integer without integer instructions. `1/3` is rounded *up*, so
  \\(e \cdot \frac13\\) never falls just short of an integer \\(e/3\\).
- \\(m \cdot 2^r \in [\frac12, 4)\\) is exact: \\(r\\) only moves the exponent.
- **Subnormals** are first scaled by \\(2^{24}\\) (`f32`) or \\(2^{54}\\)
  (`f64`), multiples of 3, and \\(k\\) is lowered by 8 or 18.

## 4. A guess

\\(\sqrt[3]{m}\\) on \\([1, 2)\\) is approximated by a polynomial in
\\(u = 2m - 3 \in [-1, 1)\\) (exact), the Chebyshev interpolant of
\\(\sqrt[3]{1.5 + u/2}\\), then multiplied by \\(\sqrt[3]{2^r}\\):

| Element | Degree | Relative error  | Bits |
|---------|--------|-----------------|------|
| `f32`   | 3      | \\(2^{-13.2}\\) | 13   |
| `f64`   | 6      | \\(2^{-21.9}\\) | 22   |

## 5. One cubically convergent step

Solving \\(t^3 = a\\) (\\(a = m \cdot 2^r\\)) with Halley's method gives

\\[
t' = t \cdot \frac{t^3 + 2a}{2t^3 + a}
   = t + t \cdot \frac{a/t^2 - t}{2t + a/t^2}
\\]

which multiplies the number of correct bits by **3**: 13 bits become 39
for `f32`, 22 become 66 for `f64`. The second form is musl's: its terms
round little.

1. **\\(t^2\\) exact.** \\(t\\) is first rounded to half the significand,
   12 bits (`f32`) or 26 (`f64`): then \\(t^2\\) fits in a float. The
   rounding is *away from zero* (multiply by \\(1 + 2^{-12}\\), then clear
   the low bits), so \\(t\\) stays on one side of the root, which musl's
   error bound relies on.
2. \\(s = a/t^2\\) rounds once; \\(s - t\\) is exact (\\(s \approx t\\),
   Sterbenz); the correction \\(c = (s - t)/(2t + s)\\) is tiny.
3. \\(t + t \cdot c\\), one multiply-add, rounds once at the result's
   scale.

musl's analysis bounds the error by \\(0.5 + 0.5/3 + \epsilon \approx 0.667\\)
ULP. Then the result is scaled by \\(2^k\\) (exact) and gets \\(x\\)'s
sign.

Without step 1's rounding, \\(t^2\\) rounds too. The error then stays
below 1 ULP on the samples tested, but loses the margin: truncating instead
of rounding away from zero already measures 0.778 ULP for `f32`.

## 6. `frexp` and `ldexp`

Two `Float` operations, written per backend with integer instructions on
the float's bits:

| Operation              | Result                              | How                                                                 |
|------------------------|-------------------------------------|---------------------------------------------------------------------|
| `x.frexp()`            | \\((m, e)\\), \\(x = m \cdot 2^e\\) | \\(m\\): keep the fraction bits, set the exponent of 1. \\(e\\): shift the exponent field down, OR it into the bits of \\(2^{23}\\) (\\(2^{52}\\)), subtract \\(2^{23} + 127\\) |
| `x.ldexp(n)`           | \\(x \cdot 2^n\\)                   | add \\(2^{23} + 127\\) to \\(n\\), shift the low bits into the exponent field: that is \\(2^n\\); multiply |

The `OR` into \\(2^{23}\\) converts the exponent to a float without an
integer-to-float instruction, which AVX2 lacks for 64-bit integers. On
AVX2 these are `vpsrld` / `vpslld` / `vpor` / `vpand`, the same with
`q` for `f64`; AVX-512 has the 512-bit forms; NEON uses `ushr`, `shl`,
`orr`, `and`. [`exp`](./exp.md) and [`ln`](./ln.md) use them too.

## 7. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                   | Fused (SIMD) | Unfused (scalar) |
|---------|------------------------------------------|--------------|------------------|
| `f32`   | all \\(2^{32}\\)                         | 0.667        | 0.667            |
| `f64`   | 300 000 samples, 200-bit reference       | 0.666        | 0.666            |

The maxima are at musl's bound, on subnormal `f32` inputs and across the
`f64` range.

## 8. Cost

Per register: `frexp`, a few multiplies and selects for the reduction, a
3- or 6-term polynomial, then two divisions and `ldexp`. On AVX2, simdly's
`cbrt` is about 3× faster than `std` for `f32` and 4× for `f64` (see
[Benchmarks](../benchmarks/README.md)).
