use std::time::Duration;

use ark_ff::UniformRand;
use criterion::*;
use dp_crypto::virtual_poly::{build_eq_x_r_vec, build_eq_x_r_vec_sequential};
use ark_std::rand::thread_rng;

criterion_group!(benches, build_eq_fn,);
criterion_main!(benches);

const NUM_SAMPLES: usize = 10;
const NV: std::ops::Range<i32> = 20..24;

type F = ark_bn254::Fq;

fn build_eq_fn(c: &mut Criterion) {
    for nv in NV {
        let mut group = c.benchmark_group(format!("build_eq_{}", nv));
        group.sample_size(NUM_SAMPLES);

        let mut rng = thread_rng();
        let r = (0..nv)
            .map(|_| F::rand(&mut rng))
            .collect::<Vec<_>>();

        group.bench_function(
            BenchmarkId::new("build_eq", format!("par_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let instant = std::time::Instant::now();
                        let _ = build_eq_x_r_vec(&r);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("build_eq", format!("seq_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let instant = std::time::Instant::now();
                        let _ = build_eq_x_r_vec_sequential(&r);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );
        group.finish();
    }
}
