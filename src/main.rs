#[derive(Debug, Copy, Clone, Default)]
struct Complex<T> {
    re: T,
    im: T,
}
impl Complex<f64> {
    fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    // e^(i*x)
    fn ei(x: f64) -> Self {
        Complex::new(x.cos(), x.sin())
    }
    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl<T> std::ops::Mul for Complex<T>
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Self) -> Self {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}
impl<T> std::ops::Add for Complex<T>
where
    T: std::ops::Add<Output = T>,
{
    type Output = Complex<T>;

    fn add(self, rhs: Self) -> Self {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}
impl<T> std::ops::Sub for Complex<T>
where
    T: std::ops::Sub<Output = T>,
{
    type Output = Complex<T>;

    fn sub(self, rhs: Self) -> Self {
        Complex {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl From<f64> for Complex<f64> {
    fn from(re: f64) -> Self {
        Complex::new(re, 0.0)
    }
}

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

fn main() {
    let signal1: Vec<f64> = (0..64)
        .into_iter()
        .map(|x| ((x as f64) / 2.0).sin())
        .collect();

    let signal2: Vec<f64> = (0..64)
        .into_iter()
        .map(|x| ((x as f64) / 5.0).sin())
        .collect();

    let signal3: Vec<f64> = signal1
        .iter()
        .zip(signal2.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    println!("signal1: {:?}", signal1);
    println!("signal2: {:?}", signal2);
    println!("signal3: {:?}", signal3);

    let output1: Vec<f64> = fft(signal1.into_iter().map(Complex::from).collect())
        .into_iter()
        .map(|z| z.abs())
        .collect();

    let output2: Vec<f64> = fft(signal2.into_iter().map(Complex::from).collect())
        .into_iter()
        .map(|z| z.abs())
        .collect();

    let output3: Vec<f64> = fft(signal3.into_iter().map(Complex::from).collect())
        .into_iter()
        .map(|z| z.abs())
        .collect();

    println!("output1: {:?}", output1);
    println!("output2: {:?}", output2);
    println!("output3: {:?}", output3);
}
