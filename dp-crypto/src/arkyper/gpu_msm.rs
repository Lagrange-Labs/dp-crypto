use std::sync::Arc;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ff::PrimeField;
use ec_gpu_gen::{G1AffineM, MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker};

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

        let bases_gpu: Vec<G1AffineM> = bases.iter().map(|p| (*p).into()).collect();
        let exps: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();

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

pub fn convert_bases_to_gpu(bases: &[G1Affine]) -> Vec<G1AffineM> {
    bases.iter().map(|p| (*p).into()).collect()
}

pub fn convert_scalars_to_bigint(scalars: &[Fr]) -> Vec<<Fr as PrimeField>::BigInt> {
    scalars.iter().map(|s| s.into_bigint()).collect()
}
