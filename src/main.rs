#[macro_use]
extern crate approx;

mod complex;
use complex::*;

fn fft(mut input: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = input.len();
    assert!(n.is_power_of_two());
    let nl = n.trailing_zeros(); // log2

    let zeta: Vec<Complex<f64>> = (0..n)
        .into_iter()
        .map(|x| Complex::ei(-2.0 * std::f64::consts::PI * (x as f64) / (n as f64)))
        .collect();

    // shuffle stage
    let shift = 64 - nl;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            input.swap(i, j);
        }
    }

    // combines the sub FFTs
    for x in 0..nl {
        let m = 2 << x;
        let m2 = 1 << x;

        for s in (0..n).step_by(m) {
            for k in 0..m2 {
                let i1 = s + k;
                let i2 = s + k + m2;
                let e = input[i1];
                let zo = zeta[(k * n / m) % n] * input[i2];
                input[i1] = e + zo;
                input[i2] = e - zo;
            }
        }
    }

    input
}

fn range<F>(steps: usize, min: f64, max: f64, f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let step = (max - min) / (steps as f64);
    (0..steps)
        .into_iter()
        .map(|s| (s as f64) * step + min)
        .map(f)
        .collect()
}

fn add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect()
}

fn to_complex_vec(a: Vec<f64>) -> Vec<Complex<f64>> {
    a.into_iter().map(Complex::from).collect()
}

fn to_abs_vec(a: Vec<Complex<f64>>) -> Vec<f64> {
    a.into_iter().map(Complex::abs).collect()
}

fn main() {
    let signal1: Vec<f64> = range(256, 0.0, 10.0, f64::sin);
    let signal2: Vec<f64> = range(256, 0.0, 18.0, f64::sin);
    let signal3: Vec<f64> = add(&signal1, &signal2);

    // println!("signal1: {:?}", signal1);
    // println!("signal2: {:?}", signal2);
    // println!("signal3: {:?}", signal3);

    let output1: Vec<f64> = to_abs_vec(fft(to_complex_vec(signal1)));
    let output2: Vec<f64> = to_abs_vec(fft(to_complex_vec(signal2)));
    let output3: &[f64] = &to_abs_vec(fft(to_complex_vec(signal3)));
    let output12: &[f64] = &add(&output1, &output2);

    // println!("output1: {:?}", output1);
    // println!("output2: {:?}", output2);
    // println!("output3: {:?}", output3);

    assert_relative_eq!(output12, output3, max_relative = 0.5);
}
