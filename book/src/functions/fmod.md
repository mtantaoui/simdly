# Remainders `fmod` and `remainder`

\\[
\operatorname{fmod}(x, y) = x - y \cdot \operatorname{trunc}(x/y), \qquad
\operatorname{remainder}(x, y) = x - y \cdot \lfloor x/y \rfloor
\\]

`fmod` is Rust's `%` and C's `fmod`: exact, with the sign of `x`.
`remainder` is Python's and PyTorch's `%`, burn's `float_remainder`, with
the sign of `y`. Each comes element-wise (`fmod(&y)`) and with one divisor
for the slice (`fmod_scalar(y)`). Results are bit for bit `%`'s and burn's
on every backend. Implementation:
[`src/math/fmod.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/fmod.rs).

## 1. Special values

| Input                         | `fmod`    | `remainder`                     |
|-------------------------------|-----------|---------------------------------|
| `y = ±0`, `x = ±∞`, NaN       | NaN       | NaN                             |
| finite `x`, `y = ±∞`          | `x`       | `x`, or `y` itself where the signs differ (`x + y`) |
| `x = ±0`                      | `±0`      | `±0`                            |

`remainder` is computed as burn computes it: \\(r = x \bmod y\\) (exact),
then \\(r + y\\) where \\(r \ne 0\\) and the signs of \\(r\\) and \\(y\\)
differ. That sum rounds, and `−0` stays `−0` (Python gives `+0`).

## 2. `fmod` as long division in SIMD lanes

No ISA has a remainder instruction, and \\(x/y\\) can reach \\(2^{2098}\\)
(`f64`): one division does not give the integer quotient. With
\\(a = \lvert x\rvert\\) and \\(b = \lvert y\rvert\\), each pass:

1. picks \\(b' = b \cdot 2^e\\) (exact) with \\(e \ge 0\\) as large as keeps
   \\(r/b' < 2^{p-1}\\), from the exponents of \\(r\\) and \\(b\\)
   (subnormals scaled first);
2. takes \\(q = \operatorname{trunc}(r/b')\\): the division rounds up at
   most to the next integer, so \\(q\\) is the integer quotient or one too
   large;
3. computes \\(r - q b'\\) exactly (`residual`: a fused multiply-add, or
   Dekker's product on the scalar backend): a multiple of \\(b'\\)'s ULP
   below \\(b'\\) in magnitude, so representable; and adds \\(b'\\) back
   where it went negative.

That leaves \\(r \bmod b'\\), and since \\(b'\\) is a multiple of \\(b\\),
\\(a \bmod b = (a \bmod b') \bmod b\\). Each pass removes \\(p - 2\\) bits of
the exponent gap: one pass while \\(a/b < 2^{p-1}\\), about 40 (`f64`) at
worst; the loop runs while some lane of the register needs it. Finally,
the sign of `x`.

Dekker's product, on the scalar backend, needs its partial products
normal: where \\(r\\) is tiny or \\(b'\\) huge, the pass runs on both
scaled by \\(2^{\pm k}\\), exactly, and the result is scaled back, exactly
too.

## 3. Measured

Bit for bit `%` and burn's `remainder_float` on 10 million random pairs
per element type, every exponent gap from none to `MAX / MIN_SUBNORMAL`,
both signs, and ±0, ±∞, NaN and subnormals on either side, on every
backend.

On AVX2 at 4096 elements, compared with `std`'s `%` (a call to `fmod`)
and burn's formula on it:

| Function                        | Inputs                                     | `f32` | `f64` |
|---------------------------------|--------------------------------------------|-------|-------|
| `fmod`                          | \\(x \in [-80, 80]\\), \\(y \in [0.5, 3.5]\\) | 2.2×  | 1.4×  |
| `remainder`                     | the same                                    | 1.9×  |       |
| `fmod_scalar(2.5)`              | \\(x \in [-80, 80]\\)                       | 2.3×  |       |
| `fmod_scalar(3)`                | \\(x \approx 10^{300}\\)                    |       | 0.29× |

With quotients near \\(2^{1000}\\), `f64` takes about 20 passes, each with a
division, where the C library shifts integers: 3.5× slower. See
[Benchmarks](../benchmarks/README.md).
