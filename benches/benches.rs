use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fft::fft_inplace;
use num_complex::ComplexDistribution;
use rand_distr::{Distribution, StandardNormal};

pub fn fft(c: &mut Criterion) {
    let dist = ComplexDistribution::new(StandardNormal, StandardNormal);
    let mut rng = rand::thread_rng();

    let mut data = dist.sample_iter(&mut rng).take(2048).collect::<Vec<_>>();
    fft_inplace(&mut data);

    for i in (4..12).rev() {
        let n = 1usize << i;

        c.bench_with_input(BenchmarkId::new("fft", n), &n, |b, n| {
            b.iter(|| fft_inplace(&mut data[..*n]));
        });
    }
}

criterion_group!(benches, fft);
criterion_main!(benches);
