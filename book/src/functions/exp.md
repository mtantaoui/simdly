# Exponential `exp`

\\[
\exp : \mathbb{R} \to (0, \infty), \qquad \exp(x) = e^x
\\]

simdly uses fdlibm's algorithm: musl's `exp.c` for `f64`, and the `expf.c`
musl shipped before its table-driven version for `f32`, which stays in
`f32` arithmetic. Implementation:
[`src/math/exp.rs`](https://github.com/mtantaoui/simdly/blob/main/src/math/exp.rs).
Error: **0.910 ULP** (`f32`) and **0.844 ULP** (`f64`) at most, on every
backend, subnormal results included ([measurements](#6-measured-error)).

## 1. Special values and range

| Input                          | Output                 |
|--------------------------------|------------------------|
| `±0`                           | `1`                    |
| `+∞`                           | `+∞`                   |
| `−∞`                           | `+0`                   |
| NaN                            | NaN                    |
| \\(x > \ln(\mathrm{MAX})\\): 88.72 (`f32`), 709.78 (`f64`) | `+∞` |
| \\(x\\) below −103.97 (`f32`), −745.13 (`f64`) | `+0`   |
| between −103.97 and −87.34 (`f32`)            | subnormal |

## 2. Range reduction

\\[
x = k \ln 2 + r, \qquad k = \operatorname{round}(x / \ln 2), \qquad \lvert r\rvert \le \tfrac{\ln 2}{2}
\\]

so \\(e^x = 2^k e^r\\). \\(k\\) is rounded with the \\(1.5 \cdot 2^{23}\\)
(\\(2^{52}\\)) trick, as in [`cbrt`](./cbrt.md). \\(\ln 2\\) is split as
`hi + lo`, where `hi` has trailing zero bits: \\(x - k \cdot \mathtt{hi}\\)
is then **exact** for every \\(k\\) in range, and
\\(r = (x - k \cdot \mathtt{hi}) - k \cdot \mathtt{lo}\\) keeps
\\(\ln 2\\) to twice the precision.

## 3. The rational approximation

fdlibm approximates not \\(e^r\\) but

\\[
R(r) = r \cdot \frac{e^r + 1}{e^r - 1} = 2 + \frac{r^2}{6} - \frac{r^4}{360} + \cdots \approx 2 + z\,P(z), \qquad z = r^2
\\]

an even function, so \\(P\\) needs few terms: 2 coefficients for `f32`
(error \\(2^{-27.7}\\)), 5 for `f64`. Solving \\(R\\) for \\(e^r\\) gives,
with \\(c = r - z\,P(z)\\),

\\[
e^r = 1 + r + \frac{r \cdot c}{2 - c}
\\]

The correction \\(\frac{r \cdot c}{2 - c}\\) is small next to \\(1 + r\\), so
its rounding errors shrink in the result. One division per register.

## 4. Evaluation order

\\[
e^r = 1 + \Bigl(\Bigl(\frac{r \cdot c}{2 - c} - \mathtt{lo}\Bigr) + \mathtt{hi}\Bigr)
\\]

with `hi` \\(= x - k \cdot \ln2_{hi}\\) and `lo` \\(= k \cdot \ln2_{lo}\\):
the smallest terms first, then the exact `hi`, and \\(1\\) last. The order
matters. Written as \\(1 + (r + \frac{r \cdot c}{2 - c})\\), which is the
same expression, the `f32` maximum goes from 0.910 to **1.003 ULP**,
measured over every input.

## 5. Scaling by \\(2^k\\)

\\(k\\) reaches 128 (`f32`) or 1024 (`f64`) near overflow, and −150 or
−1075 for subnormal results: \\(2^k\\) itself is not always a normal float.
So the result is scaled in two steps,

\\[
e^x = \bigl(e^r \cdot 2^{k_1}\bigr) \cdot 2^{k - k_1}, \qquad k_1 = \operatorname{round}(k/2)
\\]

each factor built by [`ldexp`](./cbrt.md#6-frexp-and-ldexp) and normal.
The first product is exact; the second rounds **once**, also when the
result is subnormal. Building \\(2^k\\) directly from bits, as some
implementations do, gives \\(\infty\\) at \\(k = 128\\) and garbage below
the normal range. A test with a single \\(2^k\\) returns 0 for
\\(\exp(-87.7)\\).

## 6. Measured error

Maximum error in ULPs ([how it is measured](../accuracy.md)):

| Element | Inputs                                                   | Fused (SIMD) | Unfused (scalar) |
|---------|----------------------------------------------------------|--------------|------------------|
| `f32`   | every finite input                                       | 0.910        | 0.910            |
| `f64`   | 310 000 samples, incl. both thresholds and subnormal results, 200-bit reference | 0.844 | 0.844 |

The `f64` tests use their own reference, `exp` in double-double arithmetic,
not `std`'s, which is the platform C library's (see
[`cbrt`](./cbrt.md) for why). Errors in the subnormal range are measured
in the scaled domain, so they are exact too.

For comparison, measured the same way on every `f32`:

| `f32` `exp`                         | Max error       | Inputs > 1 ULP | Wrong 0 or ∞ |
|-------------------------------------|-----------------|----------------|--------------|
| simdly                              | 0.910 ULP       | 0              | 0            |
| [erydanos](https://github.com/awxkee/erydanos) | 11.7 million ULP | 2.3 million | 94 743 |

erydanos's documented 1.0 ULP holds in the middle of the range, but it
returns \\(\infty\\) for \\(x > 88\\) (\\(e^{88.5}\\) is finite) and 0 for
\\(x < -87\\) (where results are normal or subnormal, not 0).

## 7. Cost

Per register: one compare-free reduction (two multiply-adds), a 2- or
5-term polynomial, one division, two `ldexp`, two selects for the
thresholds. On AVX2, `exp` is 5× faster than `std` for `f32` and 3.7× for
`f64` at 4096 elements (see [Benchmarks](../benchmarks/README.md)).
