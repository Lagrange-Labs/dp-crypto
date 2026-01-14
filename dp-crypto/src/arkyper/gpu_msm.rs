use std::sync::{Arc, Mutex};

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ec_gpu_gen::{G1AffineM, MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker};
use rayon::prelude::*;

pub static GPU_MSM: std::sync::LazyLock<Mutex<GpuMsm>> =
    std::sync::LazyLock::new(|| Mutex::new(GpuMsm::new().expect("Failed to initialize GPU MSM")));

pub struct GpuMsm {
    kernel: MultiexpKernel<'static, G1Affine>,
    pool: Worker,
}

impl GpuMsm {
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

    pub fn new_single_device(device_index: usize) -> anyhow::Result<Self> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No GPU devices found"));
        }
        if device_index >= devices.len() {
            return Err(anyhow::anyhow!(
                "Device index {} out of range (available: {})",
                device_index,
                devices.len()
            ));
        }

        let device = &devices[device_index];
        let program =
            program!(device).map_err(|e| anyhow::anyhow!("Failed to create GPU program: {e}"))?;

        let kernel = MultiexpKernel::create(vec![program], std::slice::from_ref(device))
            .map_err(|e| anyhow::anyhow!("Failed to create MSM kernel: {e}"))?;

        let pool = Worker::new();

        Ok(Self { kernel, pool })
    }

    pub fn msm(&mut self, bases: &[G1Affine], scalars: &[Fr]) -> anyhow::Result<G1Projective> {
        if bases.len() != scalars.len() {
            return Err(anyhow::anyhow!(
                "bases and scalars must have the same length: {} != {}",
                bases.len(),
                scalars.len()
            ));
        }

        let bases_gpu = convert_bases_to_gpu(bases);
        let exps = convert_scalars_to_bigint(scalars);

        let result = self
            .kernel
            .multiexp(&self.pool, Arc::new(bases_gpu), Arc::new(exps), 0)
            .map_err(|e| anyhow::anyhow!("GPU MSM failed: {e}"))?;

        Ok(result)
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

impl Default for GpuMsm {
    fn default() -> Self {
        Self::new().expect("Failed to create GpuMsm")
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

fn g1_to_gpu(p: &G1Affine) -> G1AffineM {
    match p.xy() {
        Some((x, y)) => G1AffineM {
            x: fq_to_montgomery_bytes(&x),
            y: fq_to_montgomery_bytes(&y),
        },
        None => G1AffineM::default(),
    }
}

pub fn convert_bases_to_gpu(bases: &[G1Affine]) -> Vec<G1AffineM> {
    bases.par_iter().map(g1_to_gpu).collect()
}

pub fn convert_scalars_to_bigint(scalars: &[Fr]) -> Vec<<Fr as PrimeField>::BigInt> {
    scalars.par_iter().map(|s| s.into_bigint()).collect()
}
