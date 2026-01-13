#[allow(unused_imports)]
use ark_bn254::{Bn254, Fr};
#[allow(unused_imports)]
use ark_poly::DenseMultilinearExtension;
#[allow(unused_imports)]
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::rand::thread_rng;
use divan::Bencher;
use dp_crypto::{
    arkyper::{CommitmentScheme, HyperKZG},
    poly::{dense::DensePolynomial as ADensePolynomial, slice::SmartSlice},
};
#[allow(unused_imports)]
use jolt_core::poly::{
    commitment::{
        commitment_scheme::CommitmentScheme as CScheme, hyperkzg::HyperKZG as JoltHyperKZG,
    },
    dense_mlpoly::DensePolynomial as JDense,
    multilinear_polynomial::MultilinearPolynomial as JMLE,
};

fn main() {
    divan::main();
}

const LENS: [usize; 3] = [20, 23, 25];

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
            let poly = ADensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(s.as_slice()));
            HyperKZG::<Bn254>::commit(&pp, &poly)
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
}

#[divan::bench_group(sample_count = 3, sample_size = 1)]
mod open {
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    #[allow(unused_imports)]
    use jolt_core::field::JoltField;
    #[allow(unused_imports)]
    use jolt_core::poly::dense_mlpoly::DensePolynomial as JoltDense;
    #[allow(unused_imports)]
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial as JoltMLE;
    #[allow(unused_imports)]
    use jolt_core::transcripts::Blake2bTranscript;
    #[allow(unused_imports)]
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
            let r_len = poly.num_vars;
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            let transcript = Blake3Transcript::new(b"hyperkzg_test");
            (pp, poly, point, transcript)
        })
        .bench_local_values(|(pp, poly, point, mut prove_transcript)| {
            HyperKZG::<Bn254>::open(&pp, &poly, &point, &Fr::ZERO, &mut prove_transcript)
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
}
