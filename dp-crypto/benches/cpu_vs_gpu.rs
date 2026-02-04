//! Focused CPU vs GPU benchmark for batch_commit and batch_open.
//!
//! 10 polynomials of 2^22 evaluations each.
//!
//! Run with:
//! ```bash
//! cargo bench --bench cpu_vs_gpu --features cuda
//! ```

use ark_bn254::{Bn254, Fr};
use ark_ff::AdditiveGroup;
use ark_std::rand::thread_rng;
use divan::Bencher;
use dp_crypto::{
    arkyper::{
        transcript::blake3::Blake3Transcript,
        CommitmentScheme, HyperKZG,
    },
    poly::dense::DensePolynomial,
};
#[cfg(feature = "cuda")]
use dp_crypto::arkyper::HyperKZGGpu;
use ark_std::rand::Rng;

fn main() {
    divan::main();
}

const LOG_N: usize = 22;
const NUM_POLYS: usize = 10;

/// Generate polynomials with small random coefficients in [0, 2^53).
/// This exercises the `compute_max_scalar_bits` optimization in GPU MSM.
fn make_polys() -> Vec<DensePolynomial<'static, Fr>> {
    let mut rng = thread_rng();
    (0..NUM_POLYS)
        .map(|_| {
            let evals: Vec<Fr> = (0..1usize << LOG_N)
                .map(|_| Fr::from(rng.gen_range(0u64..1u64 << 53)))
                .collect();
            DensePolynomial::new(evals)
        })
        .collect()
}

// ============================================================================
// Batch commit
// ============================================================================

#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod batch_commit {
    use super::*;

    #[divan::bench]
    fn cpu(b: Bencher) {
        b.with_inputs(|| {
            let polys = make_polys();
            let (pp, _) = HyperKZG::<Bn254>::test_setup(&mut thread_rng(), LOG_N);
            (pp, polys)
        })
        .bench_local_values(|(pp, polys)| {
            HyperKZG::<Bn254>::batch_commit(&pp, &polys).unwrap()
        })
    }

    #[divan::bench]
    #[cfg(feature = "cuda")]
    fn gpu(b: Bencher) {
        b.with_inputs(|| {
            let polys = make_polys();
            let (pp, _) = HyperKZGGpu::<Bn254>::test_setup(&mut thread_rng(), LOG_N);
            (pp, polys)
        })
        .bench_local_values(|(pp, polys)| {
            HyperKZGGpu::<Bn254>::batch_commit(&pp, &polys).unwrap()
        })
    }
}

// ============================================================================
// Batch open (linear-combine polys then open the combined poly)
// ============================================================================

#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod batch_open {
    use super::*;

    /// Build a single combined polynomial with small (≤53-bit) coefficients.
    fn make_open_input<CS: CommitmentScheme<Field = Fr>>() -> (CS::ProverSetup, DensePolynomial<'static, Fr>, Vec<Fr>, Blake3Transcript) {
        let polys = make_polys();
        let (pp, _) = CS::test_setup(&mut thread_rng(), LOG_N);
        let point: Vec<Fr> = (0..LOG_N).map(|i| Fr::from(i as u64)).collect();
        // Use small challenges so the linear combination stays ≤53-bit.
        let challenges: Vec<Fr> = (1..=polys.len())
            .map(|i| Fr::from(i as u64))
            .collect();
        let poly = DensePolynomial::linear_combination(
            &polys.iter().collect::<Vec<_>>(),
            &challenges,
        );
        let transcript = Blake3Transcript::new(b"bench_open");
        (pp, poly, point, transcript)
    }

    #[divan::bench]
    fn cpu(b: Bencher) {
        b.with_inputs(make_open_input::<HyperKZG<Bn254>>)
        .bench_local_values(|(pp, poly, point, mut transcript)| {
            HyperKZG::<Bn254>::open(&pp, &poly, &point, &Fr::ZERO, &mut transcript).unwrap()
        })
    }

    #[divan::bench]
    #[cfg(feature = "cuda")]
    fn gpu(b: Bencher) {
        b.with_inputs(make_open_input::<HyperKZGGpu<Bn254>>)
        .bench_local_values(|(pp, poly, point, mut transcript)| {
            HyperKZGGpu::<Bn254>::prove(&pp, &poly, &point, None, &mut transcript).unwrap()
        })
    }
}
