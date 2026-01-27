use std::sync::{Arc, Mutex};

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ec_gpu::arkworks_bn254::G1Affine as GpuG1Affine;
use ec_gpu_gen::{
    program, rust_gpu_tools::Device, threadpool::Worker, G1AffineM, MultiexpKernel,
    PolyOpsKernel,
};
use rayon::prelude::*;

pub static GPU_MSM: std::sync::LazyLock<Mutex<GpuMsm>> =
    std::sync::LazyLock::new(|| Mutex::new(GpuMsm::new().expect("Failed to initialize GPU MSM")));

/// Cached converted bases for GPU operations
struct CachedBases {
    /// Length of the original bases that were converted
    len: usize,
    /// GPU-format bases
    bases_gpu: Vec<G1AffineM>,
}

pub struct GpuMsm {
    kernel: MultiexpKernel<'static, GpuG1Affine>,
    poly_ops: PolyOpsKernel<Fr>,
    pool: Worker,
    /// Cached converted bases to avoid repeated conversions
    cached_bases: Option<CachedBases>,
}

impl GpuMsm {
    pub fn new() -> anyhow::Result<Self> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No GPU devices found"));
        }

        // Create programs for MSM kernel
        let msm_programs: Vec<_> = devices
            .iter()
            .map(|device| program!(device))
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to create GPU program for MSM: {e}"))?;

        // Create programs for poly_ops kernel
        let poly_ops_programs: Vec<_> = devices
            .iter()
            .map(|device| program!(device))
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to create GPU program for poly_ops: {e}"))?;

        let kernel = MultiexpKernel::create(msm_programs, &devices)
            .map_err(|e| anyhow::anyhow!("Failed to create MSM kernel: {e}"))?;

        let poly_ops = PolyOpsKernel::create(poly_ops_programs, &devices)
            .map_err(|e| anyhow::anyhow!("Failed to create poly_ops kernel: {e}"))?;

        let pool = Worker::new();

        Ok(Self {
            kernel,
            poly_ops,
            pool,
            cached_bases: None,
        })
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
        let msm_program =
            program!(device).map_err(|e| anyhow::anyhow!("Failed to create GPU program for MSM: {e}"))?;
        let poly_ops_program =
            program!(device).map_err(|e| anyhow::anyhow!("Failed to create GPU program for poly_ops: {e}"))?;

        let kernel = MultiexpKernel::create(vec![msm_program], std::slice::from_ref(device))
            .map_err(|e| anyhow::anyhow!("Failed to create MSM kernel: {e}"))?;

        let poly_ops = PolyOpsKernel::create(vec![poly_ops_program], &[device])
            .map_err(|e| anyhow::anyhow!("Failed to create poly_ops kernel: {e}"))?;

        let pool = Worker::new();

        Ok(Self {
            kernel,
            poly_ops,
            pool,
            cached_bases: None,
        })
    }

    /// Cache bases for repeated MSM operations with the same bases.
    ///
    /// Call this once before multiple `batch_msm_cached` calls to avoid
    /// repeated base conversions.
    pub fn cache_bases(&mut self, bases: &[G1Affine]) {
        let bases_gpu = convert_bases_to_gpu(bases);
        self.cached_bases = Some(CachedBases {
            len: bases.len(),
            bases_gpu,
        });
    }

    /// Clear cached bases.
    pub fn clear_cached_bases(&mut self) {
        self.cached_bases = None;
    }

    /// Check if bases are cached and of sufficient length.
    pub fn has_cached_bases(&self, required_len: usize) -> bool {
        self.cached_bases
            .as_ref()
            .map(|c| c.len >= required_len)
            .unwrap_or(false)
    }

    /// Get a slice of cached bases.
    ///
    /// Returns None if bases are not cached or cached length is insufficient.
    pub fn get_cached_bases_slice(&self, len: usize) -> Option<&[G1AffineM]> {
        self.cached_bases
            .as_ref()
            .filter(|c| c.len >= len)
            .map(|c| &c.bases_gpu[..len])
    }

    /// Batch MSM using cached bases (optimization #11).
    ///
    /// Requires `cache_bases` to have been called first with bases of sufficient length.
    /// Falls back to `batch_msm` if bases are not cached.
    pub fn batch_msm_cached(
        &mut self,
        bases: &[G1Affine],
        scalar_sets: &[&[Fr]],
    ) -> anyhow::Result<Vec<G1Projective>> {
        if scalar_sets.is_empty() {
            return Ok(vec![]);
        }

        let max_len = scalar_sets.iter().map(|s| s.len()).max().unwrap_or(0);

        // Check if we have cached bases of sufficient length
        let bases_gpu = if let Some(cached) = &self.cached_bases {
            if cached.len >= max_len {
                // Use cached bases
                &cached.bases_gpu[..max_len]
            } else {
                // Cache is too small, need to re-convert
                // Update the cache while we're at it
                let new_bases_gpu = convert_bases_to_gpu(&bases[..max_len]);
                self.cached_bases = Some(CachedBases {
                    len: max_len,
                    bases_gpu: new_bases_gpu,
                });
                &self.cached_bases.as_ref().unwrap().bases_gpu[..max_len]
            }
        } else {
            // No cache, convert and cache
            let new_bases_gpu = convert_bases_to_gpu(&bases[..max_len]);
            self.cached_bases = Some(CachedBases {
                len: max_len,
                bases_gpu: new_bases_gpu,
            });
            &self.cached_bases.as_ref().unwrap().bases_gpu[..max_len]
        };

        // Convert all scalar sets to bigint in parallel
        let exp_sets: Vec<Vec<_>> = scalar_sets
            .par_iter()
            .map(|scalars| {
                let mut exps = convert_scalars_to_bigint(scalars);
                exps.resize(max_len, <Fr as PrimeField>::BigInt::from(0u64));
                exps
            })
            .collect();

        // Use dynamic bit-length optimization - computes effective bits per scalar set
        self.kernel
            .batch_multiexp(bases_gpu, &exp_sets)
            .map_err(|e| anyhow::anyhow!("GPU batch MSM failed: {e}"))
    }

    /// Batch MSM using cached bases with dynamic small-scalar optimization.
    ///
    /// This variant computes the effective bit-length of scalars and only processes
    /// the windows that contain actual data. Provides significant speedup when
    /// scalars are small (e.g., 64-bit values instead of full 254-bit).
    pub fn batch_msm_cached_dynamic(
        &mut self,
        bases: &[G1Affine],
        scalar_sets: &[&[Fr]],
    ) -> anyhow::Result<Vec<G1Projective>> {
        self.batch_msm_cached(bases, scalar_sets)
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

    /// Fix the lowest variable of a multilinear polynomial on GPU.
    ///
    /// Given a polynomial in evaluation form of length 2n, computes a new polynomial
    /// of length n by fixing the lowest variable to value `r`.
    ///
    /// Formula: `out[j] = r * (poly[2j+1] - poly[2j]) + poly[2j]`
    pub fn fix_var(&self, poly: &[Fr], r: &Fr) -> anyhow::Result<Vec<Fr>> {
        self.poly_ops
            .fix_var(poly, r)
            .map_err(|e| anyhow::anyhow!("GPU fix_var failed: {e}"))
    }

    /// Fix multiple variables iteratively on GPU.
    ///
    /// Starting from a polynomial of length 2^k, fixes k variables to the given
    /// challenge values, returning the final constant value.
    pub fn fix_vars(&self, poly: &[Fr], challenges: &[Fr]) -> anyhow::Result<Fr> {
        self.poly_ops
            .fix_vars(poly, challenges)
            .map_err(|e| anyhow::anyhow!("GPU fix_vars failed: {e}"))
    }

    /// Linear combination of polynomials on GPU: out = sum(coeffs[i] * polys[i])
    ///
    /// All polynomials must have the same length.
    pub fn linear_combine(&self, polys: &[&[Fr]], coeffs: &[Fr]) -> anyhow::Result<Vec<Fr>> {
        self.poly_ops
            .linear_combine(polys, coeffs)
            .map_err(|e| anyhow::anyhow!("GPU linear_combine failed: {e}"))
    }

    /// Compute witness polynomial for KZG opening on GPU.
    ///
    /// Given f(x) and evaluation point u, computes h(x) where:
    /// f(x) = h(x) * (x - u) + f(u)
    pub fn witness_poly(&self, f: &[Fr], u: &Fr) -> anyhow::Result<Vec<Fr>> {
        self.poly_ops
            .witness_poly(f, u)
            .map_err(|e| anyhow::anyhow!("GPU witness_poly failed: {e}"))
    }

    /// Batch MSM: upload bases once, compute multiple MSMs with different scalars.
    ///
    /// More efficient than calling msm() multiple times because bases are uploaded only once.
    /// All scalar sets must be padded to the same length (use zeros for padding).
    ///
    /// Uses dynamic bit-length optimization: computes effective bit-length of scalars
    /// and only processes windows containing actual data. This provides significant
    /// speedup when scalars are small.
    pub fn batch_msm(
        &self,
        bases: &[G1Affine],
        scalar_sets: &[&[Fr]],
    ) -> anyhow::Result<Vec<G1Projective>> {
        if scalar_sets.is_empty() {
            return Ok(vec![]);
        }

        // All scalar sets should be same length for efficient batching
        let max_len = scalar_sets.iter().map(|s| s.len()).max().unwrap_or(0);

        // Convert bases to GPU format (parallelized internally)
        let bases_gpu = convert_bases_to_gpu(&bases[..max_len]);

        // Convert all scalar sets to bigint in parallel (optimization #5)
        // This parallelizes across scalar sets, and each convert_scalars_to_bigint
        // is also internally parallelized
        let exp_sets: Vec<Vec<_>> = scalar_sets
            .par_iter()
            .map(|scalars| {
                let mut exps = convert_scalars_to_bigint(scalars);
                // Pad with zeros to max_len
                exps.resize(max_len, <Fr as PrimeField>::BigInt::from(0u64));
                exps
            })
            .collect();

        // Use dynamic bit-length optimization - computes effective bits per scalar set
        // and only processes windows containing actual data
        self.kernel
            .batch_multiexp(&bases_gpu, &exp_sets)
            .map_err(|e| anyhow::anyhow!("GPU batch MSM failed: {e}"))
    }

    /// Batch MSM with fixed 254-bit assumption for BN254 scalars.
    ///
    /// Use this when scalars are known to be close to full size (254 bits)
    /// to avoid the CPU scan for effective bits.
    pub fn batch_msm_fixed_bits(
        &self,
        bases: &[G1Affine],
        scalar_sets: &[&[Fr]],
    ) -> anyhow::Result<Vec<G1Projective>> {
        if scalar_sets.is_empty() {
            return Ok(vec![]);
        }

        let max_len = scalar_sets.iter().map(|s| s.len()).max().unwrap_or(0);
        let bases_gpu = convert_bases_to_gpu(&bases[..max_len]);

        let exp_sets: Vec<Vec<_>> = scalar_sets
            .par_iter()
            .map(|scalars| {
                let mut exps = convert_scalars_to_bigint(scalars);
                exps.resize(max_len, <Fr as PrimeField>::BigInt::from(0u64));
                exps
            })
            .collect();

        const BN254_SCALAR_BITS: usize = 254;
        self.kernel
            .batch_multiexp_fixed_bits(&bases_gpu, &exp_sets, BN254_SCALAR_BITS)
            .map_err(|e| anyhow::anyhow!("GPU batch MSM failed: {e}"))
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
