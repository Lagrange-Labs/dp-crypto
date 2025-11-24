#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use std::{sync::Arc, time::Duration};

use ark_ff::{Field, PrimeField};
use ark_std::rand::thread_rng;
use criterion::*;
use dp_crypto::{
    arkyper::transcript::blake3::Blake3Transcript, monomial::Term, poly::dense::DensePolynomial,
    structs::IOPProverState, util::max_usable_threads, virtual_poly::VirtualPolynomial,
    virtual_polys::VirtualPolynomials,
};
use either::Either;
use itertools::Itertools;

criterion_group!(benches, sumcheck_fn, devirgo_sumcheck_fn,);
criterion_main!(benches);

const NUM_SAMPLES: usize = 10;
const NUM_DEGREE: usize = 3;
const NV: [usize; 2] = [25, 26];

/// transpose 2d vector without clone
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

fn prepare_input<'a, F: PrimeField>(nv: usize) -> (F, Vec<DensePolynomial<'a, F>>) {
    let mut rng = thread_rng();
    let fs = (0..NUM_DEGREE)
        .map(|_| DensePolynomial::<F>::random(nv, &mut rng))
        .collect_vec();

    let asserted_sum = fs
        .iter()
        .fold(vec![F::ONE; 1 << nv], |mut acc, f| {
            (0..f.len()).zip(acc.iter_mut()).for_each(|(i, acc)| {
                *acc *= f[i];
            });
            acc
        })
        .iter()
        .cloned()
        .sum::<F>();

    (asserted_sum, fs)
}

fn sumcheck_fn(c: &mut Criterion) {
    type F = ark_bn254::Fr;
    type T = Blake3Transcript;

    for nv in NV {
        // expand more input size once runtime is acceptable
        let mut group = c.benchmark_group(format!("sumcheck_nv_{}", nv));
        group.sample_size(NUM_SAMPLES);

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck", format!("sumcheck_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = T::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let fs = fs.into_iter().map(Arc::new).collect_vec();

                        let virtual_poly_v1 = VirtualPolynomial::new_from_product(fs, F::ONE);
                        let instant = std::time::Instant::now();
                        #[allow(deprecated)]
                        let (_sumcheck_proof_v1, _) = IOPProverState::<F>::prove_parallel(
                            virtual_poly_v1,
                            &mut prover_transcript,
                        );
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

fn devirgo_sumcheck_fn(c: &mut Criterion) {
    type F = ark_bn254::Fr;
    type T = Blake3Transcript;

    let threads = max_usable_threads();
    for nv in NV {
        // expand more input size once runtime is acceptable
        let mut group = c.benchmark_group(format!("devirgo_nv_{}", nv));
        group.sample_size(NUM_SAMPLES);

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck", format!("devirgo_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = T::new(b"test");
                        let (_, fs) = { prepare_input(nv) };

                        let virtual_poly_v2 = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: F::ONE,
                                product: fs.iter().map(Either::Left).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let (_sumcheck_proof_v2, _) =
                            IOPProverState::<F>::prove(virtual_poly_v2, &mut prover_transcript);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck_ext_in_place", format!("devirgo_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = T::new(b"test");
                        let (_, mut fs) = { prepare_input(nv) };

                        let virtual_poly_v2 = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: F::ONE,
                                product: fs.iter_mut().map(Either::Right).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let (_sumcheck_proof_v2, _) =
                            IOPProverState::<F>::prove(virtual_poly_v2, &mut prover_transcript);
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
