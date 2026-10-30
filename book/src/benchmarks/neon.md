# NEON

Speed-up of simdly's NEON backend (4 `f32` / 2 `f64` lanes) over the scalar loop; see
[Methodology](./README.md).

> **Pending:** results to come.

| Machine  |   |
|----------|---|
| CPU      | – |
| OS       | – |
| Rust     | – |
| simdly   | – |

## Arithmetic

| Operation | Type | 64 | 4 096 | 262 144 | 4 194 304 |
|---|---|---:|---:|---:|---:|
| `add` | `f32` | – | – | – | – |
| `add` | `f64` | – | – | – | – |
| `sub` | `f32` | – | – | – | – |
| `sub` | `f64` | – | – | – | – |
| `mul` | `f32` | – | – | – | – |
| `mul` | `f64` | – | – | – | – |
| `div` | `f32` | – | – | – | – |
| `div` | `f64` | – | – | – | – |
| `add_assign` | `f32` | – | – | – | – |
| `add_assign` | `f64` | – | – | – | – |
| `sub_assign` | `f32` | – | – | – | – |
| `sub_assign` | `f64` | – | – | – | – |
| `mul_assign` | `f32` | – | – | – | – |
| `mul_assign` | `f64` | – | – | – | – |
| `div_assign` | `f32` | – | – | – | – |
| `div_assign` | `f64` | – | – | – | – |

## Comparisons

| Operation | Type | 64 | 4 096 | 262 144 | 4 194 304 |
|---|---|---:|---:|---:|---:|
| `less` | `f32` | – | – | – | – |
| `less` | `f64` | – | – | – | – |
| `less_scalar` | `f32` | – | – | – | – |
| `less_scalar` | `f64` | – | – | – | – |
| `equal` | `f32` | – | – | – | – |
| `equal` | `f64` | – | – | – | – |
| `equal_scalar` | `f32` | – | – | – | – |
| `equal_scalar` | `f64` | – | – | – | – |

## Math

| Operation | Type | 64 | 4 096 | 262 144 | 4 194 304 |
|---|---|---:|---:|---:|---:|
| `abs` | `f32` | – | – | – | – |
| `abs` | `f64` | – | – | – | – |
| `abs_in_place` | `f32` | – | – | – | – |
| `abs_in_place` | `f64` | – | – | – | – |
| `sqrt` | `f32` | – | – | – | – |
| `sqrt` | `f64` | – | – | – | – |
| `sqrt_in_place` | `f32` | – | – | – | – |
| `sqrt_in_place` | `f64` | – | – | – | – |
| `acos` | `f32` | – | – | – | – |
| `acos` | `f64` | – | – | – | – |
| `asin` | `f32` | – | – | – | – |
| `asin` | `f64` | – | – | – | – |
| `atan` | `f32` | – | – | – | – |
| `atan` | `f64` | – | – | – | – |
| `atan_in_place` | `f32` | – | – | – | – |
| `atan_in_place` | `f64` | – | – | – | – |
| `atan2` | `f32` | – | – | – | – |
| `atan2` | `f64` | – | – | – | – |
| `cbrt` | `f32` | – | – | – | – |
| `cbrt` | `f64` | – | – | – | – |
| `cbrt_in_place` | `f32` | – | – | – | – |
| `cbrt_in_place` | `f64` | – | – | – | – |
| `cos` | `f32` | – | – | – | – |
| `cos` | `f64` | – | – | – | – |
| `cos_in_place` | `f32` | – | – | – | – |
| `cos_in_place` | `f64` | – | – | – | – |
| `exp` | `f32` | – | – | – | – |
| `exp` | `f64` | – | – | – | – |
| `floor` | `f32` | – | – | – | – |
| `floor` | `f64` | – | – | – | – |
| `floor_in_place` | `f32` | – | – | – | – |
| `floor_in_place` | `f64` | – | – | – | – |
| `hypot` | `f32` | – | – | – | – |
| `hypot` | `f64` | – | – | – | – |
| `ln` | `f32` | – | – | – | – |
| `ln` | `f64` | – | – | – | – |
| `sin` | `f32` | – | – | – | – |
| `sin` | `f64` | – | – | – | – |
| `sin_in_place` | `f32` | – | – | – | – |
| `sin_in_place` | `f64` | – | – | – | – |
| `tan` | `f32` | – | – | – | – |
| `tan` | `f64` | – | – | – | – |
| `tan_in_place` | `f32` | – | – | – | – |
| `tan_in_place` | `f64` | – | – | – | – |

