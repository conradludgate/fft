#![feature(slice_as_chunks)]
use std::{cell::Cell, hint::unreachable_unchecked};

use num_complex::Complex;

fn shuffle_inplace(input: &mut [Complex<f64>]) {
    let n = input.len();

    let shift = (n - 1).leading_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if j < i {
            if j >= n {
                unsafe { std::hint::unreachable_unchecked() }
            }
            input.swap(i, j);
        }
    }
}

/// combine but optimised for both 2 and 4 elements together
pub fn combine4(input: [Complex<f64>; 4]) -> [Complex<f64>; 4] {
    let [e0, o0, e1, o1] = input;
    let [e0, e1, o0, o1] = [e0 + o0, e0 - o0, e1 + o1, e1 - o1];
    let o1 = o1 * Complex::i();
    [e0 + o0, e1 + o1, e0 - o0, e1 - o1]
}

unsafe fn combine(input: &mut [Complex<f64>], zs: &[Complex<f64>]) {
    if input.len() < 2 || !input.len().is_power_of_two() || !zs.len().is_power_of_two() {
        unreachable_unchecked()
    }

    let n = input.len() / 2;
    let z_step = zs.len() / n;

    for k in 0..n {
        let z = *zs.get_unchecked(k * z_step);

        let e = *input.get_unchecked_mut(k);
        let oz = *input.get_unchecked(k + n) * z;
        *input.get_unchecked_mut(k) = e + oz;
        *input.get_unchecked_mut(k + n) = e - oz;
    }
}

/// Safety: input.len() <= 4 || !input.len().is_power_of_two() || !zs.len().is_power_of_two()
unsafe fn combine_inplace(input: &mut [Complex<f64>], zs: &[Complex<f64>]) {
    if input.len() <= 4 || !input.len().is_power_of_two() || !zs.len().is_power_of_two() {
        unreachable_unchecked()
    }

    let n = input.len();
    let x = n.trailing_zeros(); // log_2

    // fast path for groups of 2/4.
    for input in input.as_chunks_unchecked_mut() {
        *input = combine4(*input);
    }

    // handle the rest of the groupings
    for i in 3..=x {
        for j in 0..(n >> i) {
            let input =
                &mut *std::ptr::slice_from_raw_parts_mut(input.as_mut_ptr().add(j << i), 1 << i);
            combine(input, zs);
        }
    }
}

/// Performs the Cooley-Tukey Radix-2 FFT algorithm in place on power-of-two length inputs
pub fn fft_inplace(input: &mut [Complex<f64>]) {
    assert!(
        input.len().is_power_of_two(),
        "This Cooley-Tokey Radix-2 FFT implementation only works on power of two length inputs"
    );

    // skip lengths 2 and 4 since they're not very interesting
    if let [e, o] = input {
        let o1 = *e - *o;
        *e += *o;
        *o = o1;
        return;
    } else if let ([input], []) = input.as_chunks_mut() {
        input.swap(1, 2);
        *input = combine4(*input);
        return;
    }

    shuffle_inplace(input);

    let mut zs = TWIDDLES.with(|cell| cell.take());
    if zs.len() < input.len() / 2 {
        zs = twiddles(input.len());
    }

    unsafe { combine_inplace(input, &zs) };

    TWIDDLES.with(|cell| cell.set(zs));
}

thread_local! {
    static TWIDDLES: Cell<Vec<Complex<f64>>> = Cell::new(Vec::new());
}

pub fn twiddles(n: usize) -> Vec<Complex<f64>> {
    let step = -2.0 * std::f64::consts::PI / (n as f64);

    let mut zs = Vec::with_capacity(n / 2);
    for (i, z) in zs.spare_capacity_mut().iter_mut().enumerate() {
        let theta = step * (i as f64);
        z.write(Complex::from_polar(1.0, theta));
    }
    unsafe { zs.set_len(n / 2) }
    zs
}

#[test]
fn foo() {
    let mut data: Vec<Complex<f64>> = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        .into_iter()
        .map(Complex::from)
        .collect();
    fft_inplace(&mut data);
    dbg!(data);
}
