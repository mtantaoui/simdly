# Rounding, sign and clamping

Seven element-wise functions with exact results, bit for bit those of
`std` (or of burn, for `sign`), signed zeros, infinities and NaN included.
Implementations:
[`src/math/round.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/round.rs)
and
[`src/math/basic.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/basic.rs).

| Function              | Result                                                     | As                    |
|-----------------------|------------------------------------------------------------|-----------------------|
| `trunc`               | toward zero: `trunc(−2.7) = −2`, `trunc(−0.3) = −0`        | `f32::trunc`          |
| `round`               | to nearest, halves away from zero: `round(−2.5) = −3`      | `f32::round`          |
| `round_ties_even`     | to nearest, halves to even: `round_ties_even(2.5) = 2`     | `f32::round_ties_even`, burn's `round` |
| `neg`                 | `−x`, the sign bit flipped (NaN too)                       | `-x`                  |
| `recip`               | `1/x`, rounded once: `recip(±0) = ±∞`                       | `1.0 / x`             |
| `sign`                | `−1`, `0` or `1`; `0` for `±0` and NaN                      | burn's `float_sign`   |
| `clamp(min, max)`     | `min` below `min`, `max` above `max`, `x` otherwise (NaN included) | `f32::clamp`   |

`clamp` panics, as `f32::clamp`, if `min > max` or either bound is NaN;
the bounds are checked once per slice, not per element, and passed to every
register as two broadcast operands.

## Rounding

`trunc` and `round_ties_even` are one instruction per register on every
ISA: `vroundps` / `vroundpd` (AVX2) and `vrndscaleps` / `vrndscalepd`
(AVX-512) with the round-toward-zero or round-to-nearest mode, NEON's
`frintz` and `frintn`. `core` has no rounding without `std`, and `std`'s is
a library call on baseline `x86_64`, so the scalar backend rounds with
float arithmetic, as [`floor`](./floor.md): below \\(2^{p-1}\\), adding and
subtracting \\(2^{p-1}\\) rounds \\(\lvert x\rvert\\) to the nearest
integer, halves to even (and one step down gives the truncation); the sign
is put back last, so that `−0.3` gives `−0`.

`round`, halves away from zero, has no x86 rounding mode. With
\\(t = \operatorname{trunc} x\\), the dropped fraction \\(x - t\\) is
exact, and

\\[
\operatorname{round} x = \begin{cases}
t \pm 1 & \lvert x - t\rvert \ge \tfrac12 \text{, the sign of } x \\\\
t & \text{otherwise}
\end{cases}
\\]

For \\(\pm\infty\\), \\(x - t\\) is NaN and compares false, so they stay as
they are.

## Measured

Every `f32`, on the scalar and AVX2 backends, gives `std`'s bits for all
three roundings (`cargo test --release -- --ignored roundings`); `f64` and
the other functions are checked on samples over every binade, signed zeros,
infinities and NaN, on every backend.

On AVX2 at 4096 elements, compared with `std`'s scalar loop:

| Function            | `f32` | `f64` |
|---------------------|-------|-------|
| `trunc`             | 15×   | 7×    |
| `round`             | 10×   | 6×    |
| `round_ties_even`   | 24×   | 9×    |
| `neg`               | 1.0×  |       |
| `recip`             | 1.2×  | 1.0×  |
| `sign`              | 1.5×  |       |
| `clamp`             | 0.9×  | 1.0×  |

`neg`, `recip`, `sign` and `clamp` are loops that LLVM vectorizes by itself
(the comparisons of `clamp` and `sign` included), so simdly only matches
them; they are here for the same `_in_place`, `par_` and dispatch as the
rest. The roundings are where SIMD instructions make the difference: without
SSE4.1 at compile time, `std` calls a library function per element (see
[Benchmarks](../benchmarks/README.md)).
