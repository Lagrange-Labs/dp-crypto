use std::sync::{Arc, Mutex};

use ark_bn254::{Fq, Fq2, Fr, G1Projective, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;

use ec_gpu::arkworks_bn254::{G1Affine, G2Affine};
use ec_gpu_gen::{
    program, rust_gpu_tools::Device, threadpool::Worker, G1AffineM, G2AffineM, MultiexpKernel,
};
use rayon::prelude::*;

pub static GPU_MSM_G1: std::sync::LazyLock<Mutex<GpuMsmG1>> =
    std::sync::LazyLock::new(|| Mutex::new(GpuMsmG1::new().expect("Failed to initialize GPU MSM")));

pub static GPU_MSM_G2: std::sync::LazyLock<Mutex<GpuMsmG2>> =
    std::sync::LazyLock::new(|| Mutex::new(GpuMsmG2::new().expect("Failed to initialize GPU MSM")));

pub struct GpuMsmG1 {
    kernel: MultiexpKernel<'static, G1Affine>,
    pool: Worker,
}

pub struct GpuMsmG2 {
    kernel: MultiexpKernel<'static, G2Affine>,
    pool: Worker,
}

impl GpuMsmG1 {
    pub fn new() -> anyhow::Result<Self> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No GPU devices found"));
        }

        let programs: Vec<_> = devices
            .iter()
            .map(|device| program!(device))
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to create GPU program: {e}"))?;

        let kernel = MultiexpKernel::create(programs, &devices)
            .map_err(|e| anyhow::anyhow!("Failed to create MSM kernel: {e}"))?;

        let pool = Worker::new();

        Ok(Self { kernel, pool })
    }

    pub fn msm_arc(
        &mut self,
        bases: Arc<Vec<G1AffineM>>,
        scalars: Arc<Vec<<Fr as PrimeField>::BigInt>>,
    ) -> anyhow::Result<G1Projective> {
        self.kernel
            .multiexp(&self.pool, bases, scalars, 0)
            .map_err(|e| anyhow::anyhow!("GPU MSM failed: {e}"))
    }
}

impl GpuMsmG2 {
    pub fn new() -> anyhow::Result<Self> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No GPU devices found"));
        }

        let programs: Vec<_> = devices
            .iter()
            .map(|device| program!(device))
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to create GPU program: {e}"))?;

        let kernel = MultiexpKernel::create(programs, &devices)
            .map_err(|e| anyhow::anyhow!("Failed to create MSM kernel: {e}"))?;

        let pool = Worker::new();

        Ok(Self { kernel, pool })
    }

    pub fn msm_arc(
        &mut self,
        bases: Arc<Vec<G2AffineM>>,
        scalars: Arc<Vec<<Fr as PrimeField>::BigInt>>,
    ) -> anyhow::Result<G2Projective> {
        self.kernel
            .multiexp(&self.pool, bases, scalars, 0)
            .map_err(|e| anyhow::anyhow!("GPU MSM failed: {e}"))
    }
}

fn fq_to_montgomery_bytes(x: &Fq) -> [u8; 32] {
    let limbs = unsafe { std::mem::transmute::<Fq, [u64; 4]>(*x) };

    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        let bytes = limb.to_le_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    out
}

fn fq2_to_montgomery_bytes(x: &Fq2) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&fq_to_montgomery_bytes(&x.c0));
    out[32..].copy_from_slice(&fq_to_montgomery_bytes(&x.c1));
    out
}

fn g1_to_gpu(p: &ark_bn254::G1Affine) -> G1AffineM {
    match p.xy() {
        Some((x, y)) => G1AffineM {
            x: fq_to_montgomery_bytes(&x),
            y: fq_to_montgomery_bytes(&y),
        },
        None => G1AffineM::default(),
    }
}

fn g2_to_gpu(p: &ark_bn254::G2Affine) -> G2AffineM {
    match p.xy() {
        Some((x, y)) => G2AffineM {
            x: fq2_to_montgomery_bytes(&x),
            y: fq2_to_montgomery_bytes(&y),
        },
        None => G2AffineM::default(),
    }
}

pub fn convert_g1_bases_to_gpu(bases: &[ark_bn254::G1Affine]) -> Vec<G1AffineM> {
    bases.par_iter().map(g1_to_gpu).collect()
}

pub fn convert_g2_bases_to_gpu(bases: &[ark_bn254::G2Affine]) -> Vec<G2AffineM> {
    bases.par_iter().map(g2_to_gpu).collect()
}

pub fn convert_scalars_to_bigint(scalars: &[Fr]) -> Vec<<Fr as PrimeField>::BigInt> {
    scalars.par_iter().map(|s| s.into_bigint()).collect()
}
