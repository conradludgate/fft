#![feature(slice_as_chunks)]

use num_complex::Complex;

macro_rules! assume_unchecked {
    ($cond:expr $(, $($tt:tt)+)?) => {{
        if !$cond {
            // We use `if cfg!(...)` here so that any formatting arguments will be marked as used
            // by the macro, even in release mode (where they are definitely not used).
            if cfg!(debug_assertions) {
                panic!(assume_unchecked!(@panic_args $cond $(, $($tt)+)?));
            } else {
                std::hint::unreachable_unchecked()
            }
        }
    }};
    (@panic_args $cond:expr $(,)?) => { concat!("debug assertion failed: ", stringify!($cond)) };
    (@panic_args $cond:expr, $($tt:tt)+) => { $($tt)+ };
}

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
pub unsafe fn fft_inner(input: &mut [Complex<f64>], mut zs: &[Complex<f64>]) {
    let n = input.len();

    // <shuffle> - this is the recursive descent part of FFT
    let shift = n.leading_zeros() + 1;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if j < i {
            // Safety: j < i < n => j < n therefore this branch is never taken
            unsafe { assume_unchecked!(j < n) }
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
        unsafe { assume_unchecked!(len <= zs.len()) }

        let split = zs.split_at(len);
        zs = split.1;

        len <<= 1;

        // Safety: len is less than input.len(), both are powers of 2, so doubling len will never overflow
        unsafe { assume_unchecked!(len > 0) }

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
        let len = zs.len();
        zs.resize(input.len() - 4, Complex::default());

        // SAFETY: zs.len() + 4 == input.len() which is a power of two
        // and the len was less before hand and must have resulted from a prior call of this
        unsafe { twiddles(&mut zs, len) }
    }

    // SAFETY: `input.len().is_power_of_two()` and `zs.len() + 4 >= input.len()`
    unsafe { fft_inner(input, &zs) }

    TWIDDLES.with(|cell| cell.set(zs));
}

thread_local! {
    // thread local cache of the 'twiddle values'
    static TWIDDLES: std::cell::Cell<Vec<Complex<f64>>> = std::cell::Cell::new(Vec::new());
}

#[inline(never)]
/// computes the 'twiddle values' for the FFT algorithm.
/// These are e^(-2i*pi) where i is in \[0, n/2).
/// For cache efficiency, these are computed multiple times.
/// The output size of the twiddle values will be at least n-4.
/// Laid out in memory, that looks like \[\[C; 4], \[C; 8], \[C; 16], ..] but flattened.
/// We don't compute the twiddle values for n = 2/4 because they're trivially \[0] and \[0, -i] respectively.
///
/// # Safety
/// 1. zs.len() + 4 is a power of two
/// 2. len + 4 is a power of two
/// 3. len < zs.len()
unsafe fn twiddles(mut zs: &mut [Complex<f64>], len: usize) {
    // SAFETY: enforced by caller
    unsafe { assume_unchecked!(len < zs.len()) }

    // SAFETY HELPERS FOR THIS PROCEDURE:
    // len is 4 less than a power of two. Therefore the bit pattern must be something like
    // 0001_1111_1100 - a number with leading zeros, a set of ones, and then 2 zeros at the end.
    // m is defined as len + 4, so it will look like
    // 0010_0000_0000 - which has a single bit set.
    // zs.len() is also 4 less than a power of two, but strictly greater than len. Something like
    // 0011_1111_1100 - less leading zeros, but the same 2 trailing zeros.
    //
    // zs.len() - len will therefore be very simply the bitwise xor, and result in something like
    // 0010_0000_0000 - this number is clearly positive and larger than len by at least 4
    // in the above case, it's equal to m, but zs.len() can be greater, therefore m is the lower bound.
    // if zs.len() was larger, it will have more leading ones, which means that it will look like
    // 1110_0000_0000
    //
    // We start by splitting zs at len. Therefore, we have `zs.len() - len` entries in zs. (what we did above)
    // We then split that at `m`. At minimum, this leaves zs empty,
    // otherwise zs will have 1 less one digit on the end. (since m is that power of two).
    //
    // After each iteration, we double m. This makes the bit in m move over left, to match the new bit pattern of zs.len.
    // if zs is empty at the end, we exit.

    let mut m = len + 4;
    zs = zs.split_at_mut(len).1;

    loop {
        // SAFETY: See above
        unsafe { assume_unchecked!(m <= zs.len()) }

        let (section, rest) = zs.split_at_mut(m);
        zs = rest;

        // SAFETY: section.len() is m
        // m >= 4 (see above for reasoning)
        // m / 2 is therefore definitely strictly less than m, since it's non-zero
        // therefore, m == section.len() > m/2
        unsafe { assume_unchecked!(m / 2 < section.len()) }

        // the above assertion allows the bounds check to be elided here :)
        section[0].re = 1.0;
        section[m / 2].im = -1.0;

        let step = std::f64::consts::PI / (m as f64);
        for i in 1..m / 2 {
            let theta = step * (i as f64);
            let cos = theta.cos();

            // SAFETY: section.len() == m
            //       i is      in [1, m/2)
            // m/2 - i is also in [1, m/2)
            // m/2 + i is      in [m/2+1, m)
            // All of these indicies are within [0, m)
            unsafe {
                assume_unchecked!(i < section.len());
                assume_unchecked!(m / 2 - i < section.len());
                assume_unchecked!(m / 2 + i < section.len());
            }

            // the above assertion allows the bounds check to be elided here :)
            section[i].re = cos;
            section[m / 2 - i].im = -cos;
            section[m / 2 + i].im = -cos;
            section[m - i].re = -cos;
        }

        m <<= 1;
        if zs.len() == 0 {
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

#[test]
fn bench_twiddles() {
    let mut start = std::time::Instant::now();
    for _ in 0..5 {
        for _ in 0..1000000 {
            let mut v = [Complex{re: 0.0, im: 0.0}; 2048 - 4];
            unsafe { twiddles(&mut v, 0) };
        }
        let end = std::time::Instant::now();
        dbg!(end - start);
        start = end;
    }
}
