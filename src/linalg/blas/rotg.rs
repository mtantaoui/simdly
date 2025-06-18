/// Constructs the parameters for a Givens plane rotation.
///
/// Given inputs `a` (via `a_param`) and `b` (via `b_param`), this function computes
/// `c` (cosine) and `s` (sine) such that:
/// ```text
/// | c_param  s_param | | a_orig | = | r (new a_param) |
/// | -s_param c_param | | b_orig |   | 0               |
/// ```
/// The input `a_param` is overwritten with `r`, and `b_param` is overwritten with `z`,
/// where `z` is a value that can be used to reconstruct `c_param` and `s_param`.
///
/// Specifically, `z` is defined as:
/// - `s_param`, if `abs(a_original) > abs(b_original)`
/// - `1.0 / c_param`, if `abs(a_original) <= abs(b_original)` and `c_param != 0`
/// - `1.0`, if `abs(a_original) <= abs(b_original)` and `c_param == 0` (this implies `a_original == 0`)
///
/// This function mirrors the functionality of the BLAS `drotg` routine.
///
/// # Arguments
///
/// * `a_param`: On input, the value `a`. On output, this is overwritten with `r`.
/// * `b_param`: On input, the value `b`. On output, this is overwritten with `z`.
/// * `c_param`: On output, the cosine component `c` of the rotation.
/// * `s_param`: On output, the sine component `s` of the rotation.
///
/// # Examples
///
/// ```
/// ```
pub fn drotg(a_param: &mut f64, b_param: &mut f64, c_param: &mut f64, s_param: &mut f64) {
    let a_orig = *a_param;
    let b_orig = *b_param;

    let r_val: f64;
    let z_val: f64;
    let c_calc: f64;
    let s_calc: f64;

    let scale = a_orig.abs() + b_orig.abs();

    if scale == 0.0 {
        // Both a_orig and b_orig are 0.0
        c_calc = 1.0;
        s_calc = 0.0;
        r_val = 0.0;
    } else {
        // Calculate magnitude of r = sqrt(a_orig^2 + b_orig^2) using scaling
        let r_mag_component_sq = (a_orig / scale).powi(2) + (b_orig / scale).powi(2);
        let mut temp_r = scale * r_mag_component_sq.sqrt(); // This is always non-negative

        // Assign sign to temp_r. It takes the sign of a_orig if |a_orig| > |b_orig|,
        // otherwise it takes the sign of b_orig. This matches the `roe` logic.
        if a_orig.abs() > b_orig.abs() {
            temp_r = temp_r.copysign(a_orig);
        } else {
            temp_r = temp_r.copysign(b_orig);
        }

        // Handle cases where temp_r might be zero (e.g. after scaling if numbers are tiny),
        // though `scale == 0.0` should catch the a_orig=0,b_orig=0 case.
        // If temp_r is zero here, it means a_orig and b_orig were zero, caught by scale == 0.0.
        // Thus, temp_r will not be zero in this `else` block.
        c_calc = a_orig / temp_r;
        s_calc = b_orig / temp_r;
        r_val = temp_r;
    }

    // Calculate z based on the standard BLAS definition
    // (matches the C code's "Discussion" section and its effective implementation logic).
    if a_orig.abs() > b_orig.abs() {
        z_val = s_calc;
    } else {
        // Here, a_orig.abs() <= b_orig.abs()
        if c_calc == 0.0 {
            // This case implies a_orig was 0 (and b_orig was not, if scale != 0,
            // resulting in c_calc = 0 from a_orig/temp_r).
            z_val = 1.0;
        } else {
            z_val = 1.0 / c_calc;
        }
    }

    *a_param = r_val;
    *b_param = z_val;
    *c_param = c_calc;
    *s_param = s_calc;
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-9; // Tolerance for floating point comparisons

    // Helper for asserting float equality, useful for tests
    fn assert_approx_eq(val: f64, expected: f64, name: &str) {
        assert!(
            (val - expected).abs() < EPSILON,
            "{name} mismatch: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_drotg_a3_b4() {
        let mut a = 3.0;
        let mut b = 4.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        assert_approx_eq(a, 5.0, "r (a=3,b=4)");
        assert_approx_eq(b, 5.0 / 3.0, "z (a=3,b=4)");
        assert_approx_eq(c, 0.6, "c (a=3,b=4)");
        assert_approx_eq(s, 0.8, "s (a=3,b=4)");
    }

    #[test]
    fn test_drotg_a4_b3() {
        let mut a = 4.0;
        let mut b = 3.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        assert_approx_eq(a, 5.0, "r (a=4,b=3)");
        assert_approx_eq(b, 0.6, "z (a=4,b=3)"); // z = s
        assert_approx_eq(c, 0.8, "c (a=4,b=3)");
        assert_approx_eq(s, 0.6, "s (a=4,b=3)");
    }

    #[test]
    fn test_drotg_a0_b5() {
        let mut a = 0.0;
        let mut b = 5.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        assert_approx_eq(a, 5.0, "r (a=0,b=5)");
        assert_approx_eq(b, 1.0, "z (a=0,b=5)"); // z = 1.0 (c=0)
        assert_approx_eq(c, 0.0, "c (a=0,b=5)");
        assert_approx_eq(s, 1.0, "s (a=0,b=5)");
    }

    #[test]
    fn test_drotg_a5_b0() {
        let mut a = 5.0;
        let mut b = 0.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        assert_approx_eq(a, 5.0, "r (a=5,b=0)");
        assert_approx_eq(b, 0.0, "z (a=5,b=0)"); // z = s
        assert_approx_eq(c, 1.0, "c (a=5,b=0)");
        assert_approx_eq(s, 0.0, "s (a=5,b=0)");
    }

    #[test]
    fn test_drotg_a0_b0() {
        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        assert_approx_eq(a, 0.0, "r (a=0,b=0)");
        assert_approx_eq(b, 1.0, "z (a=0,b=0)"); // z = 1/c (c=1)
        assert_approx_eq(c, 1.0, "c (a=0,b=0)");
        assert_approx_eq(s, 0.0, "s (a=0,b=0)");
    }

    #[test]
    fn test_drotg_a_neg3_b4() {
        let mut a = -3.0;
        let mut b = 4.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        // roe=4. r_mag=5. r = 5.0.copysign(4.0) = 5.0
        // c = -3/5 = -0.6. s = 4/5 = 0.8
        // |a|<=|b|. c!=0. z = 1/c = 1/(-0.6) = -5/3
        assert_approx_eq(a, 5.0, "r (a=-3,b=4)");
        assert_approx_eq(b, -5.0 / 3.0, "z (a=-3,b=4)");
        assert_approx_eq(c, -0.6, "c (a=-3,b=4)");
        assert_approx_eq(s, 0.8, "s (a=-3,b=4)");
    }

    #[test]
    fn test_drotg_a3_b_neg4() {
        let mut a = 3.0;
        let mut b = -4.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        // roe=-4. r_mag=5. r = 5.0.copysign(-4.0) = -5.0
        // c = 3/(-5) = -0.6. s = -4/(-5) = 0.8
        // |a|<=|b|. c!=0. z = 1/c = 1/(-0.6) = -5/3
        assert_approx_eq(a, -5.0, "r (a=3,b=-4)");
        assert_approx_eq(b, -5.0 / 3.0, "z (a=3,b=-4)");
        assert_approx_eq(c, -0.6, "c (a=3,b=-4)");
        assert_approx_eq(s, 0.8, "s (a=3,b=-4)");
    }

    #[test]
    fn test_drotg_a_neg4_b3() {
        let mut a = -4.0;
        let mut b = 3.0;
        let mut c = 0.0;
        let mut s = 0.0;
        drotg(&mut a, &mut b, &mut c, &mut s);
        // roe=-4. r_mag=5. r = 5.0.copysign(-4.0) = -5.0
        // c = -4/(-5) = 0.8. s = 3/(-5) = -0.6
        // |a|>|b|. z = s = -0.6
        assert_approx_eq(a, -5.0, "r (a=-4,b=3)");
        assert_approx_eq(b, -0.6, "z (a=-4,b=3)");
        assert_approx_eq(c, 0.8, "c (a=-4,b=3)");
        assert_approx_eq(s, -0.6, "s (a=-4,b=3)");
    }
}
