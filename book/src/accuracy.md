# Accuracy and ULPs

A floating-point result is rarely exact; its error is measured in **ULPs**
(units in the last place): the gap between the correctly rounded result
\\(\hat y\\) and the next float away from zero.

\\[
\operatorname{err}(y) = \frac{|y - f(x)|}{\operatorname{ulp}(\hat y)}
\\]

where \\(f(x)\\) is the exact mathematical value and \\(y\\) the computed
one.

| Error          | Meaning                                                         |
|----------------|-----------------------------------------------------------------|
| ≤ 0.5 ULP      | **Correctly rounded**: \\(y = \hat y\\), the best possible float. |
| < 1 ULP        | **Faithful**: \\(y\\) is one of the two floats around \\(f(x)\\). |

`abs` and `sqrt` are correctly rounded. Transcendental functions such as
`acos` are faithful: correct rounding would need extra precision on rare
hard inputs, at a large cost for every input.

## How bounds are checked

| Element | Inputs                                   | Reference                       |
|---------|------------------------------------------|---------------------------------|
| `f32`   | all 2³² inputs, on every backend         | the `f64` result, rounded later |
| `f64`   | sampled: uniform, plus range boundaries  | `std`, and offline a 200-bit one |

For `f32`, the `f64` result has 29 more bits than needed, so the measured
error is the real one to within \\(2^{-29}\\) ULP. The exhaustive `f32`
test is ignored by default (a few minutes):

```sh
cargo test --release -- --ignored exhaustive
```

`f64` cannot be checked exhaustively. The regular tests compare every
backend against `std` on about 400 000 inputs; the maxima quoted in this
book come from a 200-bit reference ([mpmath](https://mpmath.org)) on the
same kind of sample.

`std`'s `f64` functions come from the platform's C library, and their
accuracy varies: the Windows C runtime's `cbrt` is off by 1.5 ULP on some
inputs. Where that matters, the test reference is `std`'s result corrected
by one Newton step in double-double arithmetic (about 100 bits), so the
tests measure simdly, not the platform.
