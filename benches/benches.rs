use fft::fft_inplace;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::ComplexDistribution;
use rand_distr::{StandardNormal, Distribution};

pub fn fft(c: &mut Criterion) {
    let dist = ComplexDistribution::new(StandardNormal, StandardNormal);
    let mut rng = rand::thread_rng();

    for i in 0..6 {
        let n = 64 << i;

        c.bench_function(&format!("fft_{}", n), |b| {
            let mut x = dist.sample_iter(&mut rng).take(n).collect::<Vec<_>>();
            b.iter(
                || fft_inplace(black_box(&mut x))
            )
        });
    }
}

criterion_group!(benches, fft);
criterion_main!(benches);
