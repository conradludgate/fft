#![feature(slice_as_chunks)]
use std::{cell::Cell, hint::unreachable_unchecked, time::Instant};

use num_complex::Complex;

fn shuffle_inplace(input: &mut [Complex<f64>]) {
    let n = input.len();

    let shift = n.leading_zeros() + 1;
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
fn combine4(input: [Complex<f64>; 4]) -> [Complex<f64>; 4] {
    let [e0, o0, e1, o1] = input;
    let [e0, e1, o0, o1] = [e0 + o0, e0 - o0, e1 + o1, e1 - o1];
    let o1 = o1 * -Complex::i();
    [e0 + o0, e1 + o1, e0 - o0, e1 - o1]
}

unsafe fn combine(input: &mut [Complex<f64>], zs: &[Complex<f64>]) {
    let n = input.len() / 2;
    if n < 1 || !input.len().is_power_of_two() || zs.len() != n {
        unreachable_unchecked()
    }

    for k in 0..n {
        let z = *zs.get_unchecked(k);

        let e = *input.get_unchecked_mut(k);
        let oz = *input.get_unchecked(k + n) * z;
        *input.get_unchecked_mut(k) = e + oz;
        *input.get_unchecked_mut(k + n) = e - oz;
    }
}

/// Safety: input.len() > 4 && input.len().is_power_of_two() || zs.len() + 1 == input.len()
unsafe fn combine_inplace(input: &mut [Complex<f64>], zs: &[Complex<f64>]) {
    if input.len() <= 4 || !input.len().is_power_of_two() || zs.len() + 1 != input.len() {
        unreachable_unchecked()
    }

    let n = input.len();
    let x = n.trailing_zeros(); // log_2

    // fast path for groups of 2/4.
    for input in input.as_chunks_unchecked_mut() {
        *input = combine4(*input);
    }

    // handle the rest of the groupings
    let mut z_start = 3;
    let mut z_end = 7;
    for i in 3..=x {
        let zs = unsafe { zs.get_unchecked(z_start..z_end) };
        for j in 0..(n >> i) {
            let input =
                &mut *std::ptr::slice_from_raw_parts_mut(input.as_mut_ptr().add(j << i), 1 << i);
            combine(input, zs);
        }
        z_start = z_end;
        z_end += 1 << i;
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
    zs = twiddles(zs, input.len());

    unsafe { combine_inplace(input, &zs) };

    TWIDDLES.with(|cell| cell.set(zs));
}

thread_local! {
    static TWIDDLES: Cell<Vec<Complex<f64>>> = Cell::new(Vec::new());
}

pub fn twiddles(mut zs: Vec<Complex<f64>>, n: usize) -> Vec<Complex<f64>> {
    while zs.len() + 1 < n {
        zs = twiddles_inner(zs)
    }
    zs
}

#[inline(never)]
fn twiddles_inner(mut zs: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let m = zs.len() + 1;
    let step = -std::f64::consts::PI / (m as f64);

    zs.reserve(m);
    for (i, z) in zs.spare_capacity_mut().iter_mut().enumerate().take(m) {
        let theta = step * (i as f64);
        z.write(Complex::from_polar(1.0, theta));
    }
    unsafe { zs.set_len(zs.len() + m) }
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

#[test]
fn bar() {
    let mut start = Instant::now();
    for _ in 0..5 {
        for _ in 0..1000000 {
            let mut data: Vec<Complex<f64>> = (0..2048).map(|x| Complex::from(x as f64)).collect();
            fft_inplace(&mut data);
            drop(std::hint::black_box(data));
        }
        let end = Instant::now();
        dbg!(end - start);
        start = end;
    }
}
