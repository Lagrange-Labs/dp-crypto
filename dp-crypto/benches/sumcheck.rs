use std::sync::Arc;

use ark_ff::Field;
use ark_std::rand::thread_rng;
use divan::Bencher;
use dp_crypto::{
    arkyper::transcript::blake3::Blake3Transcript, poly::dense::DensePolynomial,
    structs::IOPProverState,
};
use itertools::Itertools;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

fn prepare_input<const NUM_DEGREE: usize, MLE, Fn: FnMut(usize) -> MLE>(
    nv: usize,
    mut random_mle: Fn,
) -> Vec<MLE> {
    (0..NUM_DEGREE).map(|_| random_mle(nv)).collect_vec()
}

type F = ark_bn254::Fr;
type T = Blake3Transcript;

const NUM_VARIABLES: &[usize] = &[18, 20, 22, 24, 26];
const DEGREES: &[usize] = &[2, 3];

const SAMPLE_COUNT: u32 = 30;

#[divan::bench_group(sample_count = SAMPLE_COUNT)]
mod sumchecks {
    use std::{
        rc::Rc,
        sync::atomic::{AtomicU16, Ordering},
    };

    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_sumcheck::ml_sumcheck::{MLSumcheck, protocol::ListOfProductsOfPolynomials};
    use dp_crypto::{
        monomial::Term, util::optimal_sumcheck_threads, virtual_polys::VirtualPolynomials,
    };
    use either::Either;
    use hyperplonk::SumCheck;

    use super::*;

    #[divan::bench(args = NUM_VARIABLES, consts = DEGREES)]
    fn dp_sumcheck<const NUM_DEGREE: usize>(b: Bencher, nv: usize) {
        let num_threads = optimal_sumcheck_threads(nv);
        // we need to sample the input polynomials here because we need to provide references to
        // such polynomials to the benched function
        const NUM_INPUTS: usize = 3;
        let inputs: Vec<_> = (0..NUM_INPUTS)
            .map(|_| {
                prepare_input::<NUM_DEGREE, _, _>(nv, |n| {
                    DensePolynomial::<F>::random(n, &mut thread_rng())
                })
            })
            .collect();

        let counter = AtomicU16::new(0);
        b.with_inputs(|| {
            let num_input = counter.fetch_add(1, Ordering::Relaxed) as usize;
            let virtual_poly = VirtualPolynomials::new_from_monomials(
                num_threads,
                nv,
                vec![Term {
                    scalar: F::ONE,
                    product: inputs[num_input % NUM_INPUTS]
                        .iter()
                        .map(Either::Left)
                        .collect_vec(),
                }],
            );
            let transcript = T::new(b"dp_sumcheck");
            (virtual_poly, transcript)
        })
        .bench_values(|(virtual_poly, mut transcript)| {
            IOPProverState::prove(virtual_poly, &mut transcript);
        });
    }

    #[divan::bench(args = NUM_VARIABLES, consts = DEGREES)]
    #[cfg(feature = "nightly-benches")]
    fn ceno_sumcheck<const NUM_DEGREE: usize>(b: Bencher, nv: usize) {
        use ff_ext::GoldilocksExt2;
        use multilinear_extensions::{
            mle::MultilinearExtension, virtual_polys::VirtualPolynomials as CenoVirtualPolynomial,
        };
        use p3::field::FieldAlgebra;
        use transcript::BasicTranscript;

        type E = GoldilocksExt2;
        type T = BasicTranscript<E>;

        let num_threads = sumcheck::util::optimal_sumcheck_threads(nv);
        // we need to sample the input polynomials here because we need to provide references to
        // such polynomials to the benched function
        const NUM_INPUTS: usize = 3;
        let inputs: Vec<_> = (0..NUM_INPUTS)
            .map(|_| {
                prepare_input::<NUM_DEGREE, _, _>(nv, |n| {
                    MultilinearExtension::<E>::random(n, &mut thread_rng())
                })
            })
            .collect();

        let counter = AtomicU16::new(0);

        b.with_inputs(|| {
            let num_input = counter.fetch_add(1, Ordering::Relaxed) as usize;
            let virtual_poly = CenoVirtualPolynomial::new_from_monimials(
                num_threads,
                nv,
                vec![multilinear_extensions::monomial::Term {
                    scalar: Either::Right(E::ONE),
                    product: inputs[num_input % NUM_INPUTS]
                        .iter()
                        .map(Either::Left)
                        .collect_vec(),
                }],
            );
            let transcript = T::new(b"ceno_sumcheck");
            (virtual_poly, transcript)
        })
        .bench_local_values(|(virtual_poly, mut transcript)| {
            sumcheck::structs::IOPProverState::prove(virtual_poly, &mut transcript);
        });
    }

    #[divan::bench(args = NUM_VARIABLES, consts = DEGREES)]
    fn arkworks_sumcheck<const NUM_DEGREE: usize>(b: Bencher, nv: usize) {
        b.with_inputs(|| {
            let fs = prepare_input::<NUM_DEGREE, _, _>(nv, |n| {
                DenseMultilinearExtension::<F>::rand(n, &mut thread_rng())
            })
            .into_iter()
            .map(Rc::new)
            .collect_vec();
            let mut products = ListOfProductsOfPolynomials::new(nv);
            products.add_product(fs, F::ONE);
            products
        })
        .bench_values(|products| {
            MLSumcheck::prove(&products).unwrap();
        });
    }

    #[divan::bench(args = NUM_VARIABLES, consts = DEGREES)]
    fn hyperplonk_sumcheck<const NUM_DEGREE: usize>(b: Bencher, nv: usize) {
        b.with_inputs(|| {
            let fs = prepare_input::<NUM_DEGREE, _, _>(nv, |n| {
                DenseMultilinearExtension::<F>::rand(n, &mut thread_rng())
            })
            .into_iter()
            .map(Arc::new)
            .collect_vec();
            let mut virtual_poly = hyper_arith::VirtualPolynomial::new(nv);
            virtual_poly.add_mle_list(fs, F::ONE).unwrap();
            let transcript = hyperplonk::PolyIOP::<F>::init_transcript();
            (virtual_poly, transcript)
        })
        .bench_local_values(|(virtual_poly, mut transcript)| {
            hyperplonk::PolyIOP::prove(&virtual_poly, &mut transcript).unwrap();
        });
    }

    #[divan::bench(args = NUM_VARIABLES, consts = DEGREES)]
    #[cfg(feature = "nightly-benches")]
    fn expander_sumcheck<const NUM_DEGREE: usize>(b: Bencher, nv: usize) {
        type Fr = expander_arith::Fr;
        type T = BytesHashTranscript<expander_hasher::Keccak256hasher>;

        use expander_poly::{MultiLinearPoly, SumOfProductsPoly};
        use expander_sumcheck::transcript::{BytesHashTranscript, Transcript};

        b.with_inputs(|| {
            let mut fs = prepare_input::<NUM_DEGREE, _, _>(nv, |n| {
                MultiLinearPoly::<Fr>::random(n, &mut thread_rng())
            });
            let mut virtual_poly = SumOfProductsPoly::new();
            virtual_poly.add_pair(fs.pop().unwrap(), fs.pop().unwrap()); // we take only 2 polynomials since it supports only degree 2 sumchecks
            let transcript = T::new();
            (virtual_poly, transcript)
        })
        .bench_local_values(|(virtual_poly, mut transcript)| {
            expander_sumcheck::SumCheck::prove(&virtual_poly, &mut transcript);
        });
    }
}
