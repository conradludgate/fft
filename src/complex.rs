#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Complex<T> {
    re: T,
    im: T,
}

impl Complex<f64> {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    // e^(i*x)
    pub fn ei(x: f64) -> Self {
        Complex::new(x.cos(), x.sin())
    }
    pub fn abs(self) -> f64 {
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
