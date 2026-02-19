//! Benchmark to verify that the GPU MSM small-scalar optimization works.
//!
//! ec-gpu-gen's `compute_max_scalar_bits()` scans scalars and computes optimal
//! window count. For 53-bit scalars (typical for fix_var intermediates), this
//! should use ~6 windows vs ~26 for full 256-bit scalars, giving significant speedup.
//!
//! Run with:
//! ```bash
//! cargo bench --bench msm_bitlength --features cuda
//! ```

use ark_bn254::Fr;
use ark_std::rand::SeedableRng;
use ark_std::UniformRand;
use divan::Bencher;

fn main() {
    divan::main();
}

const SIZES: [usize; 3] = [14, 16, 18];

/// Generate scalars with at most `max_bits` significant bits.
fn generate_small_scalars(n: usize, max_bits: u32, rng: &mut impl ark_std::rand::Rng) -> Vec<Fr> {
    let mask = if max_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << max_bits) - 1
    };
    (0..n).map(|_| Fr::from(rng.next_u64() & mask)).collect()
}

#[cfg(feature = "cuda")]
#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod gpu_msm_bitlength {
    use super::*;
    use dp_crypto::arkyper::gpu_msm::{convert_bases_to_gpu, convert_scalars_to_bigint, GPU_MSM};
    use dp_crypto::arkyper::{HyperKZGSRS, HyperKZGProverKey};
    use ark_bn254::Bn254;
    use std::sync::Arc;

    /// MSM with 53-bit scalars (typical for polynomial evaluations from fix_var).
    #[divan::bench(args = SIZES)]
    fn msm_53bit_scalars(b: Bencher, log_n: usize) {
        let n = 1 << log_n;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (pk, _): (HyperKZGProverKey<Bn254>, _) = srs.trim(n);
        let bases_gpu = Arc::new(convert_bases_to_gpu(pk.g1_powers()));
        let scalars = generate_small_scalars(n, 53, &mut rng);

        b.with_inputs(|| Arc::new(convert_scalars_to_bigint(&scalars)))
            .bench_local_values(|scalars_bigint| {
                GPU_MSM
                    .lock()
                    .unwrap()
                    .msm_arc(bases_gpu.clone(), scalars_bigint)
                    .expect("GPU MSM failed")
            })
    }

    /// MSM with full 256-bit scalars.
    #[divan::bench(args = SIZES)]
    fn msm_256bit_scalars(b: Bencher, log_n: usize) {
        let n = 1 << log_n;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (pk, _): (HyperKZGProverKey<Bn254>, _) = srs.trim(n);
        let bases_gpu = Arc::new(convert_bases_to_gpu(pk.g1_powers()));
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        b.with_inputs(|| Arc::new(convert_scalars_to_bigint(&scalars)))
            .bench_local_values(|scalars_bigint| {
                GPU_MSM
                    .lock()
                    .unwrap()
                    .msm_arc(bases_gpu.clone(), scalars_bigint)
                    .expect("GPU MSM failed")
            })
    }
}

/// CPU reference benchmarks for comparison.
#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod cpu_msm_bitlength {
    use super::*;
    use ark_ec::VariableBaseMSM;
    use ark_bn254::{Bn254, G1Projective};
    use dp_crypto::arkyper::{HyperKZGSRS, HyperKZGProverKey};

    #[divan::bench(args = SIZES)]
    fn msm_53bit_scalars(b: Bencher, log_n: usize) {
        let n = 1 << log_n;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (pk, _): (HyperKZGProverKey<Bn254>, _) = srs.trim(n);
        let bases = &pk.g1_powers()[..n];
        let scalars = generate_small_scalars(n, 53, &mut rng);

        b.bench_local(|| {
            G1Projective::msm(bases, &scalars).expect("CPU MSM failed")
        })
    }

    #[divan::bench(args = SIZES)]
    fn msm_256bit_scalars(b: Bencher, log_n: usize) {
        let n = 1 << log_n;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (pk, _): (HyperKZGProverKey<Bn254>, _) = srs.trim(n);
        let bases = &pk.g1_powers()[..n];
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        b.bench_local(|| {
            G1Projective::msm(bases, &scalars).expect("CPU MSM failed")
        })
    }
}
