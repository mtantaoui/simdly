# Arc tangent `atan`

\\[
\operatorname{atan} : \mathbb{R} \to (-\tfrac{\pi}{2}, \tfrac{\pi}{2}), \qquad \tan(\operatorname{atan} x) = x
\\]

simdly uses musl's algorithm (`atanf.c`, `atan.c`, from fdlibm), with its
five range reductions merged into one formula for SIMD. Implementation:
[`src/math/atan.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/atan.rs).
Error: **< 1 ULP** for `f32` and `f64`, on every backend
([measurements](#7-measured-error)).

## 1. Special values

| Input   | Output   | How                                              |
|---------|----------|--------------------------------------------------|
| `±0`    | `±0`     | kept explicitly (the formula would give `+0`)    |
| `±1`    | `±π/4`   | `hi + lo` of `atan(1)`; nearest float            |
| `±∞`    | `±π/2`   | set explicitly ([Sec. 5](#5-the-endpoints)); nearest float |
| NaN     | NaN      | propagated                                       |

## 2. Range reduction

\\(\operatorname{atan}\\) is odd, so \\(a = \lvert x\rvert\\) is reduced and
the sign restored at the end. The subtraction formula for the tangent,
\\(\tan(u - v) = \frac{\tan u - \tan v}{1 + \tan u \tan v}\\), gives for any
\\(c\\):

\\[
\operatorname{atan}(a) = \operatorname{atan}(c) + \operatorname{atan}\frac{a - c}{1 + c\,a}
\\]

and, as \\(c \to \infty\\),
\\(\operatorname{atan}(a) = \frac{\pi}{2} + \operatorname{atan}(-\frac{1}{a})\\).
musl picks \\(c\\) per range so that the remaining argument \\(t\\) stays
small:

| Range of \\(a\\)                       | \\(c\\)  | \\(t\\)                            | \\(t\\) in              |
|----------------------------------------|----------|------------------------------------|-------------------------|
| \\([0, \frac{7}{16})\\)                | –        | \\(x\\) (signed)                   | \\((-0.44, 0.44)\\)     |
| \\([\frac{7}{16}, \frac{11}{16})\\)    | 0.5      | \\(\frac{2a - 1}{2 + a}\\)         | \\([-0.05, 0.14)\\)     |
| \\([\frac{11}{16}, \frac{19}{16})\\)   | 1        | \\(\frac{a - 1}{a + 1}\\)          | \\([-0.19, 0.09)\\)     |
| \\([\frac{19}{16}, \frac{39}{16})\\)   | 1.5      | \\(\frac{a - 1.5}{1 + 1.5a}\\)     | \\([-0.11, 0.20)\\)     |
| \\([\frac{39}{16}, \infty)\\)          | ∞        | \\(-\frac{1}{a}\\)                 | \\([-0.41, 0)\\)        |

In every range \\(\lvert t\rvert < \frac{7}{16}\\). The numerators are
**exact**: \\(2a - 1\\), \\(a - 1\\) and \\(a - 1.5\\) subtract numbers within
a factor 2 of each other (Sterbenz's lemma). Only the denominator and the
division round, and their error reaches the result scaled down by
\\(t / \operatorname{atan}(a)\\).

**One formula for SIMD.** All five reductions are

\\[
t = \frac{p\,a - q}{p + q\,a}
\\]

with \\((p, q) = (1, 0), (2, 1), (1, 1), (1, 1.5), (0, 1)\\). So a register
selects \\(p\\) and \\(q\\) per lane and divides **once**, instead of
computing five quotients. The first range uses the signed \\(x\\), so its
sign needs no restoring. Measured over every `f32`, the merged form is
exactly as accurate as musl's separate expressions (Sec. 7).

## 3. The polynomial

\\[
\operatorname{atan}(t) = t - \frac{t^3}{3} + \frac{t^5}{5} - \cdots = t - t \cdot S(t^2),
\qquad S(z) = \frac{z}{3} - \frac{z^2}{5} + \cdots
\\]

musl's coefficients are a minimax fit of \\(S\\) on
\\(\lvert t\rvert \le \frac{7}{16}\\): 5 for `f32`, 11 for `f64`. The first
are close to the series' (\\(0.33333\\), \\(-0.19999\\), …). \\(S\\) is
split by the parity of the powers of \\(w = z^2\\):

\\[
S = \underbrace{z\,(a_0 + w(a_2 + w\,a_4 + \cdots))}_{s_1} + \underbrace{w\,(a_1 + w(a_3 + \cdots))}_{s_2}
\\]

two independent Horner chains of half the length, which the CPU runs in
parallel.

## 4. Reconstruction

With \\(\operatorname{atan}(c) = \mathtt{hi} + \mathtt{lo}\\) (both zero for
the first range):

\\[
\operatorname{atan}(a) = \mathtt{hi} - \bigl((t \cdot S - \mathtt{lo}) - t\bigr)
\\]

As for [`acos`](./acos.md#5-π2-in-two-parts), the small terms combine
first, and only the last subtraction rounds at the scale of the result. On
the SIMD backends \\(t \cdot S - \mathtt{lo}\\) is one fused multiply-add.

| \\(c\\) | `f32` `hi`   | `f64` `hi`             |
|---------|--------------|------------------------|
| 0.5     | `0x3eed6338` | `0x3fddac670561bb4f`   |
| 1       | `0x3f490fda` | `0x3fe921fb54442d18`   |
| 1.5     | `0x3f7b985e` | `0x3fef730bd281f69b`   |
| ∞       | `0x3fc90fda` | `0x3ff921fb54442d18`   |

## 5. The endpoints

- **\\(\pm 0\\):** the reconstruction computes \\(0 - (\ldots)\\), which is
  \\(+0\\) for both signs, so \\(x = \pm 0\\) returns \\(x\\) itself.
- **\\(\pm\infty\\):** \\(p = 0\\) and \\(0 \cdot \infty\\) is NaN, so the
  result is set to \\(\pm(\mathtt{hi} + \mathtt{lo})\\), the float nearest
  \\(\pm\frac{\pi}{2}\\).
- **Large finite \\(a\\):** \\(t = -1/a\\) is tiny, and the result rounds
  to the float nearest \\(\frac{\pi}{2}\\), as it should.

## 6. On SIMD

Per register: four compares (\\(a\\) against \\(\frac{7}{16}\\),
\\(\frac{11}{16}\\), \\(\frac{19}{16}\\), \\(\frac{39}{16}\\)); \\(p\\),
\\(q\\), `hi` and `lo` picked with four selects each; one division; the
polynomial; then selects for the sign, \\(\pm\infty\\) and \\(\pm 0\\).

The 16 selects of constants are the price of having no lookup table in the
generic `Float` operations, and probably part of why `atan` gains less from
SIMD than `acos` (see [Benchmarks](../benchmarks/README.md)).

## 7. Measured error

Maximum error in ULPs per range of \\(a\\)
([how it is measured](../accuracy.md)). `f32`: every input; `f64`: 380 000
samples (every magnitude, and \\(\pm 10^4\\) neighbours of each boundary)
against a 200-bit reference.

| Element | Backend           | \\([0, \frac{7}{16})\\) | \\([\frac{7}{16}, \frac{11}{16})\\) | \\([\frac{11}{16}, \frac{19}{16})\\) | \\([\frac{19}{16}, \frac{39}{16})\\) | \\(\ge \frac{39}{16}\\) |
|---------|-------------------|-------|-------|-------|-------|-------|
| `f32`   | SIMD (fused)      | 0.707 | 0.843 | 0.852 | 0.701 | 0.757 |
| `f32`   | scalar (unfused)  | 0.707 | 0.843 | 0.852 | 0.701 | 0.765 |
| `f64`   | SIMD (fused)      | 0.675 | 0.836 | 0.847 | 0.699 | 0.746 |
| `f64`   | scalar (unfused)  | 0.675 | 0.836 | 0.847 | 0.699 | 0.747 |

The largest errors are in the middle ranges, around \\(a \approx 0.69\\),
where \\(t\\) is near the edge of its range.
