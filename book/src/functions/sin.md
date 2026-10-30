# Sine `sin`

\\[
\sin : \mathbb{R} \to [-1, 1]
\\]

in radians, for every finite input, \\(10^{300}\\) included. simdly uses
fdlibm's kernels and argument reduction, as in musl (`k_sin.c`, `k_cos.c`,
`k_sinf.c`, `k_cosf.c`, `e_rem_pio2.c`, `e_rem_pio2f.c`). Implementation:
[`src/math/trig.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/trig.rs).
Error: **0.501 ULP** (`f32`) and **0.774 ULP** (`f64`) at most, on every
backend ([measurements](#6-measured-error)).

## 1. Special values

| Input      | Output |
|------------|--------|
| `±0`       | `±0`   |
| `±∞`, NaN  | NaN    |

## 2. Range reduction

\\[
x = n \cdot \tfrac{\pi}{2} + y, \qquad n = \operatorname{round}\left(x \cdot \tfrac{2}{\pi}\right), \qquad \lvert y\rvert \le \tfrac{\pi}{4}
\\]

then, by \\(n \bmod 4\\):

| \\(n \bmod 4\\) | 0 | 1 | 2 | 3 |
|-----------------|---|---|---|---|
| \\(\sin x\\)    | \\(\sin y\\) | \\(\cos y\\) | \\(-\sin y\\) | \\(-\cos y\\) |

A register computes both polynomials for every lane and selects per lane.
\\(n\\) is rounded as in [`exp`](./exp.md): add and subtract
\\(1.5 \cdot 2^{52}\\).

The hard part is \\(y\\): \\(x - n \cdot \frac{\pi}{2}\\) cancels, so
\\(\frac{\pi}{2}\\) must be known to far more bits than a float holds. It is
split into parts with trailing zeros, so that each \\(n \cdot p_i\\) is
exact.

## 3. `f32`: computed in `f64`

As musl's `sinf`, the `f32` lanes are widened to `f64`, computed, and
rounded once at the end: a new `Float` operation, `via_f64`, converts a
register to two `f64` registers and back (`vcvtps2pd` / `vcvtpd2ps` on
x86, `fcvtl` / `fcvtn` on NEON). With 53 bits for a 24-bit result:

- **Reduction:** \\(\frac{\pi}{2} = p_1 + p_{1t}\\), \\(p_1\\) with 25
  bits, is enough for \\(\lvert x\rvert < 2^{28} \cdot \frac{\pi}{2}\\).
- **Kernels:** degree 9 for \\(\sin y\\), 8 for \\(\cos y\\), musl's
  `k_sinf` / `k_cosf`, accurate to \\(2^{-34}\\) or better.

The result is then within a hair of correctly rounded: 0.501 ULP measured.

## 4. `f64`: three rounds

\\(\frac{\pi}{2} = p_1 + p_2 + p_{2t} = p_1 + p_2 + p_3 + p_{3t}\\),
each \\(p_i\\) with 33 bits, as `e_rem_pio2.c`, for
\\(\lvert x\rvert < 2^{20} \cdot \frac{\pi}{2}\\):

1. \\(t = x - n p_1\\), exact.
2. \\(r = t - n p_2\\) and its rounding error are kept, and
   \\(y = y_0 + y_1\\) as two floats, with \\(n p_{2t}\\) the tail.
3. Where more than 49 bits cancelled, the same step with \\(p_3\\) and
   \\(p_{3t}\\).

Round 3 is computed for every lane and selected only where needed: applied
everywhere, it drops round 2's rounding error. Needed it is: the `f64`
input closest to a multiple of \\(\frac{\pi}{2}\\) below the limit,
\\(642615.918\ldots\\), is \\(2^{-72.6}\\) away in relative terms, and
without round 3, \\(826882.894\ldots\\) measures 1.06 ULP.

The kernels (degree 13 for \\(\sin\\), 14 for \\(\cos\\)) take
\\(y_0 + y_1\\) and use \\(y_1\\) to first order.

## 5. Huge arguments

Above those limits, a lane is reduced **exactly** (Payne–Hanek), with
integers, one lane at a time, and only in registers that hold such a lane
(a mask test: `vpmovmskb`, `kortest`, `umaxv`):

- With \\(\lvert x\rvert = m \cdot 2^e\\), \\(m\\) a 53-bit integer, the
  bits of \\(\frac{2}{\pi}\\) worth \\(2^{-k}\\) for small \\(k\\) only add
  multiples of 4 to \\(x \cdot \frac{2}{\pi}\\): they are skipped.
- The next 192 bits times \\(m\\) (a 256-bit product) give \\(n \bmod 4\\)
  and the fraction, which, times \\(\frac{\pi}{2}\\) in double-double,
  is \\(y\\).
- \\(\frac{2}{\pi}\\) is stored to 1 408 bits, enough for `f64::MAX`.

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)), against a
reference that reduces exactly as in Sec. 5 and evaluates the Taylor series
in double-double, checked against 3000-bit values:

| Element | Inputs                                                          | Max   |
|---------|-----------------------------------------------------------------|-------|
| `f32`   | 3·10⁵ samples of every binade to `f32::MAX`                     | 0.501 |
| `f64`   | 3·10⁵ samples of every binade to `f64::MAX`                     | 0.774 |
| `f64`   | the 2 910 inputs below \\(2^{20} \cdot \frac{\pi}{2}\\) within \\(2^{-62}\\) of a multiple of \\(\frac{\pi}{2}\\) (one-off run) | 0.500 |

The samples include the neighbours of \\(\frac{\pi}{4}\\),
\\(\frac{\pi}{2}\\), \\(\pi\\), the reduction limits and three of the
inputs closest to multiples of \\(\frac{\pi}{2}\\). The `f64` maximum
is near \\(\frac{\pi}{4}\\), the kernels' own error. Fused (SIMD) and
unfused (scalar) give the same maxima. An ignored test checks every `f32`
against 1 ULP (`cargo test --release -- --ignored sin_f32_exhaustive`).

## 7. Cost

Per register: the reduction (two or six multiply-adds), both polynomials,
the quadrant selects, and for `f32` the conversions. Lanes with huge
arguments cost a scalar call each. On AVX2, `sin` is about 3× faster than
`std` for `f32` and 2.2× for `f64` (see
[Benchmarks](../benchmarks/README.md)).
