/// Computes modulo for f32
pub fn eabsf(x: f32) -> f32 {
    f32::from_bits(0x_7fff_ffff & x.to_bits())
}

fn main() {
    println!("{}", eabsf(5.4))
}
