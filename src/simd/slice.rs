#[inline(always)]
pub fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
