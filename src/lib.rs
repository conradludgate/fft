#![feature(slice_as_chunks, slice_split_at_unchecked)]

use num_complex::Complex;

/// combine but optimised for both 2 and 4 elements together
fn combine4(input: [Complex<f64>; 4]) -> [Complex<f64>; 4] {
    let [e0, o0, e1, o1] = input;
    let [e0, e1, o0, o1] = [e0 + o0, e0 - o0, e1 + o1, e1 - o1];
    let o1 = o1 * -Complex::i();
    [e0 + o0, e1 + o1, e0 - o0, e1 - o1]
}

fn combine(input: &mut [Complex<f64>], zs: &[Complex<f64>]) {
    let (evens, odds) = input.split_at_mut(input.len() >> 1);

    for ((z, e), o) in zs.iter().copied().zip(evens).zip(odds) {
        let oz = *o * z;
        let o1 = *e - oz;
        *e += oz;
        *o = o1;
    }
}

/// # Safety:
/// `input.len().is_power_of_two()` and `zs.len() + 4 >= input.len()`
#[inline(never)]
unsafe fn fft_inner(input: &mut [Complex<f64>], mut zs: &[Complex<f64>]) {
    let n = input.len();

    // <shuffle> - this is the recursive descent part of FFT
    let shift = n.leading_zeros() + 1;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if j < i {
            if j >= n {
                // Safety: j < i < n => j < n therefore this branch is never taken
                unsafe { std::hint::unreachable_unchecked() }
            }
            input.swap(i, j);
        }
    }
    // </shuffle>

    // <combine> - this is the recursive ascent part of FFT
    // fast path for groups of 2/4.
    for input in input.as_chunks_mut().0 {
        *input = combine4(*input);
    }

    // handle the rest of the groupings
    let mut len = 4;
    while len < input.len() {
        // Safety: The sum of powers of two up to n is < n. eg 1 + 2 + 4 = 7 < 8
        // we don't count the 1,2 cases, so we skip 3. A safety requirement of this
        // function is that zs.len() + 4 >= input.len(). In the example above, that's 4 + 4 >= 8.
        // Since this split_at acts as a sum, we know it will never exceed the length of zs
        let split = unsafe { zs.split_at_unchecked(len) };
        zs = split.1;

        len <<= 1;

        // Safety: len is less than input.len(), both are powers of 2, so doubling len will never overflow
        if len == 0 {
            unsafe { std::hint::unreachable_unchecked() }
        }

        for input in input.chunks_exact_mut(len) {
            combine(input, split.0);
        }
    }
    // </combine>
}

/// Performs the Cooley-Tukey Radix-2 FFT algorithm in place on power-of-two length inputs
pub fn fft_inplace(input: &mut [Complex<f64>]) {
    assert!(
        input.len().is_power_of_two(),
        "This Cooley-Tokey Radix-2 FFT implementation only works on power of two length inputs"
    );

    // skip lengths 2 since they don't work in our fft_inner method
    if let [e, o] = input {
        let o1 = *e - *o;
        *e += *o;
        *o = o1;
        return;
    }

    let mut zs = TWIDDLES.with(|cell| cell.take());
    // load in the twiddle values if not enough
    if zs.len() + 4 < input.len() {
        twiddles(&mut zs, input.len());
    }

    // Safety: `input.len().is_power_of_two()` and `zs.len() + 4 >= input.len()`
    unsafe { fft_inner(input, &zs) }

    TWIDDLES.with(|cell| cell.set(zs));
}

thread_local! {
    // thread local cache of the 'twiddle values'
    static TWIDDLES: std::cell::Cell<Vec<Complex<f64>>> = std::cell::Cell::new(Vec::new());
}

#[inline(never)]
// computes the 'twiddle values' for the FFT algorithm.
// These are e^(-2i*pi) where i is in [0, n/2).
// For cache efficiency, these are computed multiple times.
// The output size of the twiddle values will be at least n-4.
// Laid out in memory, that looks like [[C; 4], [C; 8], [C; 16], ..] but flattened.
// We don't compute the twiddle values for n = 2/4 because they're trivially [0] and [0, -i] respectively.
fn twiddles(zs: &mut Vec<Complex<f64>>, n: usize) {
    let mut len = zs.len();

    // if we're calling this function, then we don't have enough values in zs. this only happens for n > 4
    zs.resize(n - 4, Complex::default());

    loop {
        let m = len + 4;
        let step = std::f64::consts::PI / (m as f64);

        zs[len] = Complex { re: 1.0, im: 0.0 };
        zs[len + m / 2] = Complex { re: 0.0, im: -1.0 };

        for i in 1..m / 2 {
            let theta = step * (i as f64);
            let cos = theta.cos();

            zs[len + i].re = cos;
            zs[len + i + m / 2].im = -cos;
            zs[len + m / 2 - i].im = -cos;
            zs[len + m - i].re = -cos;
        }

        len += m;
        if len + 4 >= n {
            dbg!(zs);
            return;
        }
    }
}

#[test]
fn foo() {
    let mut data: Vec<Complex<f64>> = (0..8).map(|x| Complex::from(x as f64)).collect();
    fft_inplace(&mut data);
    dbg!(data);

    let mut data: Vec<Complex<f64>> = (0..16).map(|x| Complex::from(x as f64)).collect();
    fft_inplace(&mut data);
    dbg!(data);

    let mut data: Vec<Complex<f64>> = (0..8).map(|x| Complex::from(x as f64)).collect();
    fft_inplace(&mut data);
    dbg!(data);
}

#[test]
fn bar() {
    let mut start = std::time::Instant::now();
    for _ in 0..5 {
        for _ in 0..1000000 {
            let mut data: Vec<Complex<f64>> = (0..2048).map(|x| Complex::from(x as f64)).collect();
            fft_inplace(&mut data);
            drop(std::hint::black_box(data));
        }
        let end = std::time::Instant::now();
        dbg!(end - start);
        start = end;
    }
}
