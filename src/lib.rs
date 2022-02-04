#![feature(slice_swap_unchecked)]

use num_complex::Complex;

fn combine(zs: &[Complex<f64>], input: &mut [Complex<f64>]) {
    let n = input.len();
    let z_step = zs.len() / n;

    for k in 0..n / 2 {
        unsafe {
            let z = *zs.get_unchecked(k * z_step);

            let e = *input.get_unchecked(k);
            let o = *input.get_unchecked(k + n / 2);
            *input.get_unchecked_mut(k) = e + z * o;
            *input.get_unchecked_mut(k + n / 2) = e - z * o;
        }
    }
}

pub fn fft_inplace(input: &mut [Complex<f64>]) {
    let n = input.len();

    let shift = (n - 1).leading_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            unsafe { input.swap_unchecked(i, j) };
        }
    }

    let step = -2.0 * std::f64::consts::PI / (n as f64);
    let mut theta = 0.0;

    let mut zs = Vec::with_capacity(n / 2);
    for _ in 0..n / 2 {
        zs.push(Complex::from_polar(1.0, theta));
        theta += step;
    }

    let mut m = 2;
    while m <= n {
        for input in input.chunks_exact_mut(m) {
            combine(&zs, input);
        }
        m <<= 1;
    }
}
