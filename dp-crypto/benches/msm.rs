use std::sync::Arc;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, scalar_mul::variable_base::VariableBaseMSM};
use ark_std::{UniformRand, test_rng};
use divan::Bencher;
use dp_crypto::arkyper::gpu_msm::{GpuMsm, convert_bases_to_gpu, convert_scalars_to_bigint};

fn main() {
    divan::main();
}

const SIZES: [usize; 5] = [12, 14, 16, 18, 20];

fn generate_random_bases(n: usize) -> Vec<G1Affine> {
    let mut rng = test_rng();
    (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect()
}

fn generate_random_scalars(n: usize) -> Vec<Fr> {
    let mut rng = test_rng();
    (0..n).map(|_| Fr::rand(&mut rng)).collect()
}

#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod cpu_msm {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn arkworks_msm(b: Bencher, n: usize) {
        let size = 1 << n;
        b.with_inputs(|| {
            let bases = generate_random_bases(size);
            let scalars = generate_random_scalars(size);
            (bases, scalars)
        })
        .bench_local_values(|(bases, scalars)| {
            <G1Projective as VariableBaseMSM>::msm(&bases, &scalars).unwrap()
        })
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1)]
mod gpu_msm_bench {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn gpu_msm(b: Bencher, n: usize) {
        let size = 1 << n;
        let mut gpu = GpuMsm::new().expect("Failed to create GPU MSM");

        b.with_inputs(|| {
            let bases = generate_random_bases(size);
            let scalars = generate_random_scalars(size);
            (bases, scalars)
        })
        .bench_local_values(|(bases, scalars)| gpu.msm(&bases, &scalars).unwrap())
    }

    #[divan::bench(args = SIZES)]
    fn gpu_msm_preconverted(b: Bencher, n: usize) {
        let size = 1 << n;
        let mut gpu = GpuMsm::new().expect("Failed to create GPU MSM");

        b.with_inputs(|| {
            let bases = generate_random_bases(size);
            let scalars = generate_random_scalars(size);
            let bases_gpu = Arc::new(convert_bases_to_gpu(&bases));
            let scalars_bigint = Arc::new(convert_scalars_to_bigint(&scalars));
            (bases_gpu, scalars_bigint)
        })
        .bench_local_values(|(bases_gpu, scalars_bigint)| {
            gpu.msm_arc(bases_gpu, scalars_bigint).unwrap()
        })
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1)]
mod conversion_overhead {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn bases_conversion(b: Bencher, n: usize) {
        let size = 1 << n;
        b.with_inputs(|| generate_random_bases(size))
            .bench_local_values(|bases| convert_bases_to_gpu(&bases))
    }

    #[divan::bench(args = SIZES)]
    fn scalars_conversion(b: Bencher, n: usize) {
        let size = 1 << n;
        b.with_inputs(|| generate_random_scalars(size))
            .bench_local_values(|scalars| convert_scalars_to_bigint(&scalars))
    }
}
