use ark_bn254::{Bn254, Fr};
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::rand::thread_rng;
use divan::Bencher;
use dp_crypto::{
    arkyper::{CommitmentScheme, HyperKZG},
    poly::{dense::DensePolynomial as ADensePolynomial, slice::SmartSlice},
};
use jolt_core::poly::{
    commitment::{
        commitment_scheme::CommitmentScheme as CScheme, hyperkzg::HyperKZG as JoltHyperKZG,
    },
    dense_mlpoly::DensePolynomial as JDense,
    multilinear_polynomial::MultilinearPolynomial as JMLE,
};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const LENS: [usize; 3] = [12, 14, 16];

const NUM_BATCHED_POLYS: [usize; 2] = [3, 5];

// Register a `fibonacci` function and benchmark it over multiple cases.
#[divan::bench_group(sample_count = 3, sample_size = 1)]
mod commit {
    use super::*;

    fn arkworks_static_evals(n: usize) -> Vec<Fr> {
        (0..n).map(|i| Fr::from(i as u64)).collect()
    }

    #[divan::bench(args = LENS)]
    fn arkyper_commit(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let evals = arkworks_static_evals(2u32.pow(n as u32) as usize);
            (evals, HyperKZG::<Bn254>::test_setup(&mut thread_rng(), n))
        })
        .bench_local_values(|(s, (pp, _))| {
            let poly = ADensePolynomial::new(s);
            HyperKZG::<Bn254>::commit(&pp, &poly)
        })
    }

    #[divan::bench(args = LENS, consts = NUM_BATCHED_POLYS)]
    fn arkyper_batch_commit<const NUM_BATCHED_POLYS: usize>(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let polys = (0..NUM_BATCHED_POLYS)
                .map(|_| ADensePolynomial::new(arkworks_static_evals(2u32.pow(n as u32) as usize)))
                .collect::<Vec<_>>();
            (polys, HyperKZG::<Bn254>::test_setup(&mut thread_rng(), n))
        })
        .bench_local_values(|(polys, (pp, _))| {
            HyperKZG::<Bn254>::batch_commit(&pp, &polys).unwrap()
        })
    }

    #[divan::bench(args = LENS)]
    fn arkworks_commit(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let values = arkworks_static_evals(2u32.pow(n as u32) as usize);
            let up = MultilinearPC::<Bn254>::setup(n, &mut thread_rng());
            (values, MultilinearPC::trim(&up, n))
        })
        .bench_local_values(|(s, (pk, _))| {
            let poly = DenseMultilinearExtension::from_evaluations_slice(
                s.len().ilog2() as usize,
                s.as_slice(),
            );
            MultilinearPC::<Bn254>::commit(&pk, &poly)
        })
    }

    #[divan::bench(args = LENS)]
    fn jolt_hyperkzg_commit(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let evals = arkworks_static_evals(2u32.pow(n as u32) as usize);
            (evals, JoltHyperKZG::setup_prover(n))
        })
        .bench_local_values(|(s, setup)| {
            let poly = JMLE::LargeScalars(JDense::new(s));
            JoltHyperKZG::<Bn254>::commit(&setup, &poly)
        })
    }

    #[divan::bench(args = LENS)]
    #[cfg(feature = "nightly-benches")]
    fn basefold_commit(b: Bencher, n: usize) {
        use ff_ext::GoldilocksExt2;
        use mpcs::{Basefold, BasefoldRSParams, PolynomialCommitmentScheme};
        use multilinear_extensions::{mle::MultilinearExtension, util::transpose};
        use witness::RowMajorMatrix;

        type E = GoldilocksExt2;
        type Pcs = Basefold<E, BasefoldRSParams>;

        b.with_inputs(|| {
            let mle = MultilinearExtension::<E>::random(n, &mut thread_rng());
            let poly_size = mle.evaluations.len();
            let (pp, _) = {
                Pcs::trim(
                    Pcs::setup(poly_size, mpcs::SecurityLevel::Conjecture100bits).unwrap(),
                    poly_size,
                )
                .unwrap()
            };
            let rmm = RowMajorMatrix::new_by_inner_matrix(
                p3::matrix::dense::DenseMatrix::new(
                    transpose(vec![mle.get_base_field_vec().to_vec()]).concat(),
                    1,
                ),
                witness::InstancePaddingStrategy::Default,
            );
            (pp, rmm)
        })
        .bench_local_values(|(pp, rmm)| Pcs::batch_commit(&pp, vec![rmm]).unwrap());
    }

    #[divan::bench(args = LENS, consts = NUM_BATCHED_POLYS)]
    #[cfg(feature = "nightly-benches")]
    fn basefold_batch_commit<const NUM_BATCHED_POLYS: usize>(b: Bencher, n: usize) {
        use ff_ext::GoldilocksExt2;
        use mpcs::{Basefold, BasefoldRSParams, PolynomialCommitmentScheme};
        use multilinear_extensions::{mle::MultilinearExtension, util::transpose};
        use witness::RowMajorMatrix;

        type E = GoldilocksExt2;
        type Pcs = Basefold<E, BasefoldRSParams>;

        b.with_inputs(|| {
            let mles = (0..NUM_BATCHED_POLYS)
                .map(|_| MultilinearExtension::<E>::random(n, &mut thread_rng()))
                .collect::<Vec<_>>();
            let poly_size = 1 << n;
            let (pp, _) = {
                Pcs::trim(
                    Pcs::setup(poly_size, mpcs::SecurityLevel::Conjecture100bits).unwrap(),
                    poly_size,
                )
                .unwrap()
            };
            let rmm = RowMajorMatrix::new_by_inner_matrix(
                p3::matrix::dense::DenseMatrix::new(
                    transpose(
                        mles.iter()
                            .map(|mle| mle.get_base_field_vec().to_vec())
                            .collect(),
                    )
                    .concat(),
                    mles.len(),
                ),
                witness::InstancePaddingStrategy::Default,
            );
            (pp, rmm)
        })
        .bench_local_values(|(pp, rmm)| Pcs::batch_commit(&pp, vec![rmm]).unwrap());
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1)]
mod open {
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;
    use dp_crypto::arkyper::transcript::Transcript;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use jolt_core::field::JoltField;
    use jolt_core::poly::dense_mlpoly::DensePolynomial as JoltDense;
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial as JoltMLE;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript as T;

    use super::*;

    fn arkworks_static_evals(n: usize) -> Vec<Fr> {
        (0..n).map(|i| Fr::from(i as u64)).collect()
    }

    #[divan::bench(args = LENS)]
    fn arkyper_open(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let evals = arkworks_static_evals(2u32.pow(n as u32) as usize);
            let (pp, _) = HyperKZG::<Bn254>::test_setup(&mut thread_rng(), n);
            let poly = ADensePolynomial::new_from_smart_slice(SmartSlice::Owned(evals));
            let r_len = poly.num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            let transcript = Blake3Transcript::new(b"hyperkzg_test");
            (pp, poly, point, transcript)
        })
        .bench_local_values(|(pp, poly, point, mut prove_transcript)| {
            HyperKZG::<Bn254>::open(&pp, &poly, &point, &Fr::ZERO, &mut prove_transcript)
        })
    }

    #[divan::bench(args = LENS, consts = NUM_BATCHED_POLYS)]
    fn arkyper_batch_open<const NUM_BATCHED_POLYS: usize>(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let polys = (0..NUM_BATCHED_POLYS)
                .map(|_| ADensePolynomial::new(arkworks_static_evals(2u32.pow(n as u32) as usize)))
                .collect::<Vec<_>>();
            let (pp, _) = HyperKZG::<Bn254>::test_setup(&mut thread_rng(), n);
            let point = (0..n).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            let transcript = Blake3Transcript::new(b"hyperkzg_test");
            (pp, polys, point, transcript)
        })
        .bench_local_values(|(pp, polys, point, mut prove_transcript)| {
            let challenges = (0..polys.len())
                .map(|_| prove_transcript.challenge_scalar())
                .collect::<Vec<_>>();
            let poly = ADensePolynomial::linear_combination(
                &polys.iter().collect::<Vec<_>>(),
                &challenges,
            );
            HyperKZG::<Bn254>::open(&pp, &poly, &point, &Fr::ZERO, &mut prove_transcript).unwrap()
        })
    }

    #[divan::bench(args = LENS)]
    fn arkworks_multilinear_open(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let values = arkworks_static_evals(2u32.pow(n as u32) as usize);
            let up = MultilinearPC::<Bn254>::setup(n, &mut thread_rng());
            let (pk, _) = MultilinearPC::trim(&up, n);
            let poly = DenseMultilinearExtension::from_evaluations_slice(n, values.as_slice());
            let r_len = poly.num_vars;
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (pk, poly, point)
        })
        .bench_local_values(|(pk, poly, point)| MultilinearPC::<Bn254>::open(&pk, &poly, &point))
    }
    #[divan::bench(args = LENS)]
    fn jolt_hyperkzg_open(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let evals = arkworks_static_evals(2u32.pow(n as u32) as usize);
            let pp = JoltHyperKZG::setup_prover(n);
            let poly = JoltMLE::LargeScalars(JoltDense::new(evals));
            let r_len = poly.get_num_vars();
            let point = (0..r_len)
                .map(|i| <Fr as JoltField>::Challenge::from(i as u128))
                .collect::<Vec<_>>();
            let transcript = Blake2bTranscript::new(b"hyperkzg_test");
            (pp, poly, point, transcript)
        })
        .bench_local_values(|(pp, poly, point, mut prove_transcript)| {
            JoltHyperKZG::<Bn254>::open(&pp, &poly, &point, &Fr::ZERO, &mut prove_transcript)
        })
    }

    #[divan::bench(args = LENS)]
    #[cfg(feature = "nightly-benches")]
    fn basefold_open(b: Bencher, n: usize) {
        use ff_ext::{FromUniformBytes, GoldilocksExt2};
        use mpcs::{Basefold, BasefoldRSParams, PolynomialCommitmentScheme};
        use multilinear_extensions::{mle::MultilinearExtension, util::transpose};
        use transcript::BasicTranscript;
        use witness::RowMajorMatrix;

        type E = GoldilocksExt2;
        type T = BasicTranscript<E>;
        type Pcs = Basefold<E, BasefoldRSParams>;

        b.with_inputs(|| {
            let mle = MultilinearExtension::<E>::random(n, &mut thread_rng());
            let poly_size = mle.evaluations.len();
            let (pp, _) = {
                Pcs::trim(
                    Pcs::setup(poly_size, mpcs::SecurityLevel::Conjecture100bits).unwrap(),
                    poly_size,
                )
                .unwrap()
            };
            let r_len = mle.num_vars();
            let point = (0..r_len)
                .map(|_| E::random(&mut thread_rng()))
                .collect::<Vec<_>>();
            let eval = mle.evaluate(&point);
            let transcript = T::new(b"basefold_bench");
            let rmm = RowMajorMatrix::new_by_inner_matrix(
                p3::matrix::dense::DenseMatrix::new(
                    transpose(vec![mle.get_base_field_vec().to_vec()]).concat(),
                    1,
                ),
                witness::InstancePaddingStrategy::Default,
            );
            let commitment = Pcs::batch_commit(&pp, vec![rmm]).unwrap();
            (pp, point, eval, commitment, transcript)
        })
        .bench_local_values(|(pp, point, eval, commitment, mut prove_transcript)| {
            Pcs::batch_open(
                &pp,
                vec![(&commitment, vec![(point, vec![eval])])],
                &mut prove_transcript,
            )
            .unwrap()
        });
    }

    #[divan::bench(args = LENS, consts = NUM_BATCHED_POLYS)]
    #[cfg(feature = "nightly-benches")]
    fn basefold_batch_open<const NUM_BATCHED_POLYS: usize>(b: Bencher, n: usize) {
        use ff_ext::{FromUniformBytes, GoldilocksExt2};
        use mpcs::{Basefold, BasefoldRSParams, PolynomialCommitmentScheme};
        use multilinear_extensions::{mle::MultilinearExtension, util::transpose};
        use transcript::BasicTranscript;
        use witness::RowMajorMatrix;

        type E = GoldilocksExt2;
        type T = BasicTranscript<E>;
        type Pcs = Basefold<E, BasefoldRSParams>;

        b.with_inputs(|| {
            let mles = (0..NUM_BATCHED_POLYS)
                .map(|_| MultilinearExtension::<E>::random(n, &mut thread_rng()))
                .collect::<Vec<_>>();
            let poly_size = 1 << n;
            let (pp, _) = {
                Pcs::trim(
                    Pcs::setup(poly_size, mpcs::SecurityLevel::Conjecture100bits).unwrap(),
                    poly_size,
                )
                .unwrap()
            };
            let r_len = n;
            let point = (0..r_len)
                .map(|_| E::random(&mut thread_rng()))
                .collect::<Vec<_>>();
            let transcript = T::new(b"basefold_bench");
            let rmm = RowMajorMatrix::new_by_inner_matrix(
                p3::matrix::dense::DenseMatrix::new(
                    transpose(
                        mles.iter()
                            .map(|mle| mle.get_base_field_vec().to_vec())
                            .collect(),
                    )
                    .concat(),
                    mles.len(),
                ),
                witness::InstancePaddingStrategy::Default,
            );
            let commitment = Pcs::batch_commit(&pp, vec![rmm]).unwrap();
            let evals = mles
                .iter()
                .map(|mle| mle.evaluate(&point))
                .collect::<Vec<_>>();
            (pp, point, evals, commitment, transcript)
        })
        .bench_local_values(|(pp, point, evals, commitment, mut prove_transcript)| {
            Pcs::batch_open(
                &pp,
                vec![(&commitment, vec![(point, evals)])],
                &mut prove_transcript,
            )
            .unwrap()
        });
    }
}
