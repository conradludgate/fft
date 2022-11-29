use std::hint::black_box;

use num_complex::Complex;

fn main() {
    let mut data: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0); 1024];
    for i in 0..1024 {
        fft::fft_inplace(&mut data);
        data = black_box(data);
    }
}
