//! GPU implementation of HyperKZG polynomial commitment scheme.
//!
//! This module provides a GPU-accelerated implementation of HyperKZG that
//! implements the `CommitmentScheme` trait, allowing direct comparison with
//! the CPU implementation.
//!
//! Key features:
//! - Batch commit: Single GPU call for list of polynomials → list of commitments
//! - Batch open: GPU-accelerated polynomial operations (fix_var, linear_combine)
//! - Full trait implementation for CPU/GPU comparison

use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine};
use ark_ec::{CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::AdditiveGroup;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::Mul;

use super::gpu_msm::{convert_bases_from_gpu, convert_bases_to_gpu};
use super::transcript::Transcript;
use super::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey,
    kzg_open_batch,
};
use crate::arkyper::interface::CommitmentScheme;
use crate::poly::dense::DensePolynomial;
use ec_gpu::arkworks_bn254::G1Affine as GpuG1Affine;
use ec_gpu_gen::{
    FusedPolyCommit, G1AffineM, Phase3Input, PolyOpsKernel, compute_work_units, program,
    rust_gpu_tools::Device,
};

/// Evaluate a polynomial (given as a coefficient slice) at a point using Horner's method.
/// Equivalent to `DensePolynomial::eval_as_univariate` but operates on `&[F]` directly,
/// avoiding the need to wrap data in a `DensePolynomial`.
fn eval_as_univariate(coeffs: &[Fr], r: &Fr) -> Fr {
    let mut output = coeffs[0];
    let mut rpow = *r;
    for z in coeffs.iter().skip(1) {
        output += rpow * z;
        rpow *= r;
    }
    output
}

/// GPU-accelerated HyperKZG implementation.
///
/// This struct provides GPU-accelerated versions of commit and open operations
/// that can be directly compared with the CPU implementation.
#[derive(Clone, Debug)]
pub struct HyperKZGGpu<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> Default for HyperKZGGpu<P> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<P: Pairing> HyperKZGGpu<P> {
    pub fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
// GPU-Native SRS and Prover Key Types
// ============================================================================

/// GPU-native SRS for HyperKZG.
///
/// Stores SRS bases in GPU Montgomery format (`Vec<G1AffineM>`), eliminating
/// the CPU↔GPU format roundtrip when used with GPU operations.
pub struct HyperKZGGpuSRS {
    /// Powers of g in GPU Montgomery format: [g, τ·g, τ²·g, ..., τⁿ·g]
    powers_of_g_gpu: Vec<G1AffineM>,
    /// First power of g in CPU affine format (for verifier key)
    g: G1Affine,
    /// Single gamma_g point in CPU affine format (for verifier key)
    gamma_g: G1Affine,
    /// G2 generator
    h: G2Affine,
    /// beta * h in G2
    beta_h: G2Affine,
    /// Precomputed pairing data for h
    prepared_h: <Bn254 as Pairing>::G2Prepared,
    /// Precomputed pairing data for beta_h
    prepared_beta_h: <Bn254 as Pairing>::G2Prepared,
}

impl HyperKZGGpuSRS {
    /// Trim the SRS to a maximum degree, producing a GPU prover key and verifier key.
    pub fn trim(
        mut self,
        mut max_degree: usize,
    ) -> (HyperKZGGpuProverKey, HyperKZGVerifierKey<Bn254>) {
        if max_degree == 1 {
            max_degree += 1;
        }
        self.powers_of_g_gpu
            .resize(max_degree + 1, G1AffineM::default());

        let pk = HyperKZGGpuProverKey {
            bases_gpu: self.powers_of_g_gpu,
        };

        let vk = HyperKZGVerifierKey {
            kzg_vk: ark_poly_commit::kzg10::VerifierKey {
                g: self.g,
                gamma_g: self.gamma_g,
                h: self.h,
                beta_h: self.beta_h,
                prepared_h: self.prepared_h,
                prepared_beta_h: self.prepared_beta_h,
            },
        };

        (pk, vk)
    }

    /// Convert a CPU-format SRS to GPU-native format.
    pub fn from_cpu(srs: HyperKZGSRS<Bn254>) -> Self {
        let params = srs.0;
        let powers_of_g_gpu = convert_bases_to_gpu(&params.powers_of_g);

        Self {
            powers_of_g_gpu,
            g: params.powers_of_g[0],
            gamma_g: params.powers_of_gamma_g[&0],
            h: params.h,
            beta_h: params.beta_h,
            prepared_h: params.prepared_h,
            prepared_beta_h: params.prepared_beta_h,
        }
    }
}

/// GPU-native prover key for HyperKZG.
///
/// Stores SRS bases in GPU Montgomery format, ready for direct upload to GPU
/// without conversion overhead.
#[derive(Clone, Debug)]
pub struct HyperKZGGpuProverKey {
    bases_gpu: Vec<G1AffineM>,
}

impl HyperKZGGpuProverKey {
    pub fn bases_gpu(&self) -> &[G1AffineM] {
        &self.bases_gpu
    }

    /// Convert a CPU-format prover key to GPU-native format.
    pub fn from_cpu(pk: &HyperKZGProverKey<Bn254>) -> Self {
        Self {
            bases_gpu: convert_bases_to_gpu(pk.g1_powers()),
        }
    }
}

impl Serialize for HyperKZGGpuProverKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let n = self.bases_gpu.len();
        let mut data = vec![0u8; n * 64];
        for (i, m) in self.bases_gpu.iter().enumerate() {
            data[i * 64..i * 64 + 32].copy_from_slice(&m.x);
            data[i * 64 + 32..(i + 1) * 64].copy_from_slice(&m.y);
        }
        (n, data).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for HyperKZGGpuProverKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (n, data): (usize, Vec<u8>) = Deserialize::deserialize(deserializer)?;
        if data.len() != n * 64 {
            return Err(serde::de::Error::custom(format!(
                "invalid data length: expected {} bytes for {} bases, got {}",
                n * 64,
                n,
                data.len()
            )));
        }
        let mut bases_gpu = Vec::with_capacity(n);
        for i in 0..n {
            let mut x = [0u8; 32];
            let mut y = [0u8; 32];
            x.copy_from_slice(&data[i * 64..i * 64 + 32]);
            y.copy_from_slice(&data[i * 64 + 32..(i + 1) * 64]);
            bases_gpu.push(G1AffineM { x, y });
        }
        Ok(HyperKZGGpuProverKey { bases_gpu })
    }
}

/// GPU kernel holder for polynomial operations.
/// This is lazily initialized when first needed.
pub struct GpuPolyOpsHolder {
    poly_ops: Option<PolyOpsKernel<Fr>>,
}

impl GpuPolyOpsHolder {
    pub fn new() -> Self {
        Self { poly_ops: None }
    }

    pub fn get_or_init(&mut self) -> anyhow::Result<&PolyOpsKernel<Fr>> {
        if self.poly_ops.is_none() {
            let devices = Device::all();
            if devices.is_empty() {
                return Err(anyhow::anyhow!("No GPU devices found"));
            }

            let programs: Vec<_> = devices
                .iter()
                .map(|device| program!(device))
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to create GPU program: {e}"))?;

            let poly_ops = PolyOpsKernel::create(programs, &devices)
                .map_err(|e| anyhow::anyhow!("Failed to create poly_ops kernel: {e}"))?;

            self.poly_ops = Some(poly_ops);
        }

        Ok(self.poly_ops.as_ref().unwrap())
    }
}

pub static GPU_POLY_OPS: std::sync::LazyLock<std::sync::Mutex<GpuPolyOpsHolder>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(GpuPolyOpsHolder::new()));

// ============================================================================
// GPU Batch Operations
// ============================================================================

/// Threshold below which CPU MSM is faster than GPU (avoids kernel launch overhead).
/// Matches `GPU_MSM_THRESHOLD` in ec-gpu-gen/src/gpu_buffer.rs.
const GPU_MSM_THRESHOLD: usize = 4096;

/// Batch commit using GPU - groups polynomials by size for optimal performance.
///
/// Instead of zero-padding all polys to the max length (which wastes ~79x compute
/// for mixed-size workloads), this groups polys by their actual size:
/// - Tiny polys (≤ GPU_MSM_THRESHOLD): CPU MSM via rayon (avoids GPU kernel overhead)
/// - GPU-sized polys: one `batch_commit` call per size group with correctly-sized bases
pub fn gpu_batch_commit(
    bases_gpu: &[G1AffineM],
    polys: &[&DensePolynomial<Fr>],
) -> anyhow::Result<Vec<G1Projective>> {
    let _span = tracing::debug_span!("gpu_batch_commit", n_polys = polys.len()).entered();
    if polys.is_empty() {
        return Ok(vec![]);
    }

    let mut results = vec![G1Projective::ZERO; polys.len()];

    // Group polys by length, preserving original indices for result placement
    let mut by_size: std::collections::BTreeMap<usize, Vec<(usize, &[Fr])>> =
        std::collections::BTreeMap::new();
    for (i, p) in polys.iter().enumerate() {
        by_size.entry(p.len()).or_default().push((i, p.evals_ref()));
    }

    let max_len = polys.iter().map(|p| p.len()).max().unwrap();
    let has_tiny = by_size.keys().any(|&len| len <= GPU_MSM_THRESHOLD);

    let mut guard = GPU_FUSED.lock().unwrap();

    // Cache CPU bases only if we have tiny polys that need CPU fallback
    if has_tiny {
        guard.ensure_cpu_bases_cached(bases_gpu);
    }

    // Upload bases for the largest group; persistent buffer covers all smaller groups
    guard.ensure_bases_uploaded_gpu(&bases_gpu[..max_len])?;

    // All mutations done — take immutable references for the loop
    let holder = &*guard;
    let fused = holder.fused.as_ref().unwrap();
    let cpu_bases = holder.cached_cpu_bases.as_ref().map(|(_, v)| v.as_slice());

    for (&poly_len, group) in &by_size {
        if poly_len <= GPU_MSM_THRESHOLD {
            // CPU fallback: rayon parallel across polys
            let bases = cpu_bases.unwrap();
            let cpu_results: Vec<(usize, G1Projective)> = {
                use rayon::prelude::*;
                group
                    .par_iter()
                    .map(|(orig_idx, evals)| {
                        let r = <G1Projective as VariableBaseMSM>::msm(
                            &bases[..evals.len()],
                            evals,
                        )
                        .expect("CPU MSM failed");
                        (*orig_idx, r)
                    })
                    .collect()
            };
            for (idx, r) in cpu_results {
                results[idx] = r;
            }
        } else {
            // GPU: Pippenger MSM pipeline per poly
            let slices: Vec<&[Fr]> = group.iter().map(|(_, e)| *e).collect();
            let group_results = fused
                .batch_commit(&slices, &bases_gpu[..poly_len])
                .map_err(|e| anyhow::anyhow!("GPU batch commit error: {e}"))?;
            for ((orig_idx, _), result) in group.iter().zip(group_results) {
                results[*orig_idx] = result;
            }
        }
    }

    Ok(results)
}

/// GPU-accelerated fix_var operation.
///
/// Fixes the lowest variable of a multilinear polynomial:
/// out[j] = r * (poly[2j+1] - poly[2j]) + poly[2j]
pub fn gpu_fix_var(poly: &[Fr], r: &Fr) -> anyhow::Result<Vec<Fr>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .fix_var(poly, r)
        .map_err(|e| anyhow::anyhow!("GPU fix_var error: {e}"))
}

/// GPU-accelerated fix_vars_with_intermediates.
///
/// Fixes multiple variables and returns all intermediate polynomials.
/// This is the key operation for HyperKZG Phase 1.
pub fn gpu_fix_vars_with_intermediates(
    poly: &[Fr],
    challenges: &[Fr],
) -> anyhow::Result<Vec<Vec<Fr>>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .kernel()
        .fix_vars_with_intermediates(poly, challenges)
        .map_err(|e| anyhow::anyhow!("GPU fix_vars_with_intermediates error: {e}"))
}

/// GPU-accelerated linear combination of polynomials.
pub fn gpu_linear_combine(polys: &[&[Fr]], coeffs: &[Fr]) -> anyhow::Result<Vec<Fr>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .linear_combine(polys, coeffs)
        .map_err(|e| anyhow::anyhow!("GPU linear_combine error: {e}"))
}

/// GPU-accelerated witness polynomial computation.
pub fn gpu_witness_poly(f: &[Fr], u: &Fr) -> anyhow::Result<Vec<Fr>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .witness_poly(f, u)
        .map_err(|e| anyhow::anyhow!("GPU witness_poly error: {e}"))
}

// ============================================================================
// CPU Reference Operations (for comparison)
// ============================================================================

/// CPU reference implementation for fix_var.
pub fn cpu_fix_var(poly: &[Fr], r: &Fr) -> Vec<Fr> {
    let n = poly.len() / 2;
    (0..n)
        .map(|j| {
            let low = poly[2 * j];
            let high = poly[2 * j + 1];
            *r * (high - low) + low
        })
        .collect()
}

/// CPU reference implementation for fix_vars_with_intermediates.
pub fn cpu_fix_vars_with_intermediates(poly: &[Fr], challenges: &[Fr]) -> Vec<Vec<Fr>> {
    let mut results = Vec::with_capacity(challenges.len());
    let mut current = poly.to_vec();

    for r in challenges {
        current = cpu_fix_var(&current, r);
        results.push(current.clone());
    }

    results
}

/// CPU batch commit.
pub fn cpu_batch_commit(
    g1_powers: &[G1Affine],
    polys: &[&DensePolynomial<Fr>],
) -> anyhow::Result<Vec<G1Projective>> {
    polys
        .iter()
        .map(|poly| {
            let coeffs = poly.evals_ref();
            let msm_size = coeffs.len();
            G1Projective::msm(&g1_powers[..msm_size], coeffs)
                .map_err(|e| anyhow::anyhow!("CPU MSM error: {e}"))
        })
        .collect()
}

// ============================================================================
// HyperKZG GPU Fused Open Implementation
// ============================================================================

/// GPU kernel holder for fused poly-commit operations.
/// This is lazily initialized when first needed.
pub struct GpuFusedHolder {
    fused: Option<FusedPolyCommit<Fr, GpuG1Affine>>,
    /// Cached GPU-format bases converted from CPU-format (for backward compat path).
    cached_bases_gpu: Option<(usize, Vec<G1AffineM>)>,
    /// Cached CPU-format bases reverse-converted from GPU-format (for hybrid MSM in fused_open).
    cached_cpu_bases: Option<(usize, Vec<G1Affine>)>,
}

impl GpuFusedHolder {
    pub fn new() -> Self {
        Self {
            fused: None,
            cached_bases_gpu: None,
            cached_cpu_bases: None,
        }
    }

    pub fn get_or_convert_bases(&mut self, bases: &[G1Affine]) -> &[G1AffineM] {
        if self
            .cached_bases_gpu
            .as_ref()
            .map_or(true, |(len, _)| *len != bases.len())
        {
            let _span =
                tracing::debug_span!("convert_bases_to_gpu", n_bases = bases.len()).entered();
            self.cached_bases_gpu = Some((bases.len(), convert_bases_to_gpu(bases)));
        }
        &self.cached_bases_gpu.as_ref().unwrap().1
    }

    pub fn get_or_init(&mut self) -> anyhow::Result<&FusedPolyCommit<Fr, GpuG1Affine>> {
        self.ensure_init()?;
        Ok(self.fused.as_ref().unwrap())
    }

    fn ensure_init(&mut self) -> anyhow::Result<()> {
        if self.fused.is_none() {
            let devices = Device::all();
            if devices.is_empty() {
                return Err(anyhow::anyhow!("No GPU devices found"));
            }

            let device = &devices[0];
            let prog = program!(device)
                .map_err(|e| anyhow::anyhow!("Failed to create GPU program: {e}"))?;
            let wu = compute_work_units(device);

            let fused = FusedPolyCommit::create(prog, wu, device.memory())
                .map_err(|e| anyhow::anyhow!("Failed to create FusedPolyCommit: {e}"))?;

            self.fused = Some(fused);
        }
        Ok(())
    }

    /// Upload GPU-format bases directly to the persistent GPU buffer.
    /// No CPU→GPU format conversion needed.
    pub fn ensure_bases_uploaded_gpu(&mut self, bases_gpu: &[G1AffineM]) -> anyhow::Result<()> {
        self.ensure_init()?;
        self.fused
            .as_mut()
            .unwrap()
            .upload_bases(bases_gpu)
            .map_err(|e| anyhow::anyhow!("Failed to upload bases to GPU: {e}"))
    }

    /// Ensure CPU bases cache is populated for the given GPU bases.
    /// This is needed for fused_open's hybrid MSM which falls back to CPU for small MSMs.
    fn ensure_cpu_bases_cached(&mut self, bases_gpu: &[G1AffineM]) {
        if self
            .cached_cpu_bases
            .as_ref()
            .map_or(true, |(len, _)| *len != bases_gpu.len())
        {
            let _span =
                tracing::debug_span!("convert_bases_from_gpu", n_bases = bases_gpu.len()).entered();
            self.cached_cpu_bases = Some((bases_gpu.len(), convert_bases_from_gpu(bases_gpu)));
        }
    }

    /// Ensure SRS bases are uploaded to GPU as a persistent buffer (CPU-format input path).
    ///
    /// Converts bases to GPU format (cached), then uploads to GPU memory once.
    /// Subsequent calls with the same-sized bases are no-ops.
    pub fn ensure_bases_uploaded(&mut self, bases: &[G1Affine]) -> anyhow::Result<()> {
        self.ensure_init()?;
        let _ = self.get_or_convert_bases(bases);
        let cached_bases = &self.cached_bases_gpu.as_ref().unwrap().1;
        self.fused
            .as_mut()
            .unwrap()
            .upload_bases(cached_bases)
            .map_err(|e| anyhow::anyhow!("Failed to upload bases to GPU: {e}"))
    }
}

pub static GPU_FUSED: std::sync::LazyLock<std::sync::Mutex<GpuFusedHolder>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(GpuFusedHolder::new()));

// ============================================================================
// GPU-Accelerated SRS Setup
// ============================================================================

/// GPU-accelerated SRS (trusted setup) generation.
///
/// Computes `[G, tau*G, tau^2*G, ..., tau^n*G]` using GPU batch scalar
/// multiplication, returning a GPU-native SRS with bases already in
/// GPU Montgomery format.
///
/// Unlike the CPU `HyperKZGSRS::setup`, this skips computing the full
/// `powers_of_gamma_g` array — only the single `gamma_g` point is kept
/// for the verifier key, since GPU operations don't use gamma_g powers.
///
/// # Arguments
/// * `rng` - Random number generator for generating tau, gamma, and base points
/// * `max_degree` - Maximum polynomial degree (number of G1 powers = max_degree + 1)
///
/// # Returns
/// `HyperKZGGpuSRS` containing GPU-native bases and CPU-format verifier components.
pub fn gpu_setup<R: ark_std::rand::Rng + ark_std::rand::RngCore>(
    rng: &mut R,
    max_degree: usize,
) -> anyhow::Result<HyperKZGGpuSRS> {
    use ark_ec::CurveGroup;
    use ark_ff::{One, UniformRand};

    if max_degree < 1 {
        return Err(anyhow::anyhow!("Degree must be at least 1"));
    }

    let _span = tracing::debug_span!("gpu_setup", max_degree).entered();

    // Generate random secrets (same as arkworks KZG10::setup)
    let beta = Fr::rand(rng); // tau in our notation
    let g = G1Projective::rand(rng);
    let gamma_g = G1Projective::rand(rng);
    let h = ark_bn254::G2Projective::rand(rng);

    // Compute powers of beta: [1, beta, beta^2, ..., beta^max_degree]
    let powers_of_beta_span = tracing::debug_span!("compute_powers_of_beta").entered();
    let mut powers_of_beta = vec![Fr::one()];
    let mut cur = beta;
    for _ in 0..max_degree {
        powers_of_beta.push(cur);
        cur *= &beta;
    }
    drop(powers_of_beta_span);

    // Convert base point to GPU format
    let g_affine: G1Affine = g.into_affine();
    let g_gpu: GpuG1Affine = GpuG1Affine::from(g_affine);

    // GPU batch scalar multiplication for powers_of_g
    let powers_of_g_span = tracing::debug_span!("gpu_batch_scalar_mul_powers_of_g").entered();
    let powers_of_g: Vec<G1Affine> = GPU_FUSED
        .lock()
        .unwrap()
        .get_or_init()?
        .batch_scalar_mul(&g_gpu, &powers_of_beta)
        .map_err(|e| anyhow::anyhow!("GPU batch_scalar_mul error: {e}"))?;
    drop(powers_of_g_span);

    // Convert to GPU Montgomery format immediately
    let convert_span = tracing::debug_span!("convert_powers_to_gpu").entered();
    let powers_of_g_gpu = convert_bases_to_gpu(&powers_of_g);
    drop(convert_span);

    // Single gamma_g point for verifier key (no need for full powers_of_gamma_g)
    let gamma_g_affine: G1Affine = gamma_g.into_affine();

    // G2 computations (small, done on CPU)
    let h_affine = h.into_affine();
    let beta_h = h.into_affine().mul(beta).into_affine();
    let prepared_h = h_affine.into();
    let prepared_beta_h = beta_h.into();

    Ok(HyperKZGGpuSRS {
        powers_of_g_gpu,
        g: g_affine,
        gamma_g: gamma_g_affine,
        h: h_affine,
        beta_h,
        prepared_h,
        prepared_beta_h,
    })
}

impl HyperKZGGpu<Bn254> {
    /// GPU-accelerated open operation using fused GPU session.
    ///
    /// Runs the entire HyperKZG open in a single GPU session:
    /// - Bases uploaded directly from GPU-native prover key (no format conversion)
    /// - Scalar conversion happens on GPU (no CPU `convert_scalars_to_bigint`)
    /// - Witness polynomials never leave GPU
    /// - CPU transcript work runs inside the GPU session via callback
    pub fn open_gpu<ProofTranscript: Transcript>(
        pk: &HyperKZGGpuProverKey,
        poly: &DensePolynomial<Fr>,
        point: &[Fr],
        _eval: &Fr,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<HyperKZGProof<Bn254>> {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell);

        let _span = tracing::debug_span!("open_gpu::fused", ell, n).entered();
        let open_gpu_start = std::time::Instant::now();

        let challenges = &point[..ell - 1];

        // Keep a reference to poly evals for the callback (no clone needed)
        let poly_evals = poly.evals_ref();

        // Single lock scope: upload GPU bases directly (no conversion), then run fused_open
        let result = {
            let mut guard = GPU_FUSED.lock().unwrap();
            let upload_start = std::time::Instant::now();
            guard.ensure_bases_uploaded_gpu(pk.bases_gpu())?;
            eprintln!(
                "[open_gpu] ensure_bases_uploaded_gpu: {:?}",
                upload_start.elapsed()
            );

            // Ensure CPU bases cache is populated for fused_open's hybrid MSM
            // (small MSM fallback uses CPU bases). The &mut borrow ends here.
            guard.ensure_cpu_bases_cached(pk.bases_gpu());

            // Now take immutable borrows on separate fields
            let fused = guard.fused.as_ref().unwrap();
            let cpu_bases = &guard.cached_cpu_bases.as_ref().unwrap().1;
            fused
                .fused_open(
                    poly.evals_ref(),
                    challenges,
                    pk.bases_gpu(),
                    cpu_bases,
                    |intermediates, commitments| {
                        // === CPU work inside the GPU session ===
                        let callback_detail_start = std::time::Instant::now();

                        // Build slice references to all polynomials (original + intermediates)
                        // Avoids cloning into DensePolynomial wrappers.
                        let mut poly_slices: Vec<&[Fr]> = Vec::with_capacity(ell);
                        poly_slices.push(poly_evals);
                        for interm in intermediates {
                            poly_slices.push(interm.as_slice());
                        }

                        assert_eq!(poly_slices.len(), ell);
                        assert_eq!(poly_slices[ell - 1].len(), 2);

                        // Convert commitments to affine for transcript
                        let coms_aff: Vec<G1Affine> =
                            commitments.iter().map(|c| c.into_affine()).collect();

                        // Transcript: append intermediate commitments and get challenge
                        transcript.append_points(&coms_aff);
                        let r: Fr = transcript.challenge_scalar();
                        let u = vec![r, -r, r * r];

                        eprintln!(
                            "[open_gpu]   callback transcript: {:?}",
                            callback_detail_start.elapsed()
                        );
                        let eval_start = std::time::Instant::now();

                        // Evaluate f_i(u_j) on CPU using Horner's method on raw slices,
                        // parallelized with rayon for large polynomial counts.
                        let k = poly_slices.len();
                        let t = u.len();
                        let mut v = vec![vec![Fr::ZERO; k]; t];
                        {
                            use rayon::prelude::*;
                            v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
                                v_i.par_iter_mut().zip(poly_slices.par_iter()).for_each(
                                    |(v_ij, coeffs)| {
                                        *v_ij = eval_as_univariate(coeffs, &u[i]);
                                    },
                                );
                            });
                        }

                        eprintln!(
                            "[open_gpu]   callback evals ({}x{}): {:?}",
                            t,
                            k,
                            eval_start.elapsed()
                        );

                        // Transcript: append evals and get q_powers
                        let scalars: Vec<&Fr> = v.iter().flatten().collect();
                        transcript.append_scalars::<Fr>(&scalars);
                        let q_powers: Vec<Fr> = transcript.challenge_scalar_powers(k);

                        Phase3Input {
                            lc_coeffs: q_powers,
                            eval_points: u,
                            intermediate_commitments_affine: coms_aff,
                            evaluations: v,
                        }
                    },
                )
                .map_err(|e| anyhow::anyhow!("GPU fused_open error: {e}"))?
        };

        eprintln!(
            "[open_gpu] fused_open returned: {:?}",
            open_gpu_start.elapsed()
        );

        // Final transcript work (witness commitments)
        let final_transcript_start = std::time::Instant::now();
        let w_aff: Vec<G1Affine> = result
            .witness_commitments
            .iter()
            .map(|g| g.into_affine())
            .collect();
        transcript.append_points(&w_aff);
        let _d_0: Fr = transcript.challenge_scalar();
        eprintln!(
            "[open_gpu] final transcript: {:?}",
            final_transcript_start.elapsed()
        );
        eprintln!("[open_gpu] TOTAL: {:?}", open_gpu_start.elapsed());

        Ok(HyperKZGProof {
            coms: result.intermediate_commitments_affine,
            w: w_aff,
            v: result.evaluations,
        })
    }

    /// CPU reference open operation (for comparison).
    pub fn open_cpu<ProofTranscript: Transcript>(
        pk: &HyperKZGProverKey<Bn254>,
        poly: &DensePolynomial<Fr>,
        point: &[Fr],
        _eval: &Fr,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<HyperKZGProof<Bn254>> {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell);

        // Phase 1: Create commitments using CPU fix_var
        let mut polys: Vec<DensePolynomial<Fr>> = Vec::with_capacity(ell);
        polys.push(poly.clone());

        for i in 0..ell - 1 {
            let previous_poly = &polys[i];
            let coeffs = cpu_fix_var(previous_poly.evals_ref(), &point[i]);
            polys.push(DensePolynomial::new(coeffs));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // Commit using CPU MSM
        let poly_refs: Vec<&DensePolynomial<Fr>> = polys[1..].iter().collect();
        let coms = cpu_batch_commit(pk.g1_powers(), &poly_refs)?;
        let coms_aff: Vec<G1Affine> = coms.iter().map(|c| c.into_affine()).collect();

        // Phase 2
        transcript.append_points(&coms_aff);
        let r: Fr = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3
        let (w, v) = kzg_open_batch(&polys, &u, pk, transcript)?;

        Ok(HyperKZGProof {
            coms: coms_aff,
            w,
            v,
        })
    }
}

// ============================================================================
// CommitmentScheme Implementation for GPU
// ============================================================================

impl CommitmentScheme for HyperKZGGpu<Bn254> {
    type Field = Fr;
    type ProverSetup = HyperKZGGpuProverKey;
    type VerifierSetup = HyperKZGVerifierKey<Bn254>;
    type Commitment = HyperKZGCommitment<Bn254>;
    type Proof = HyperKZGProof<Bn254>;
    type BatchedProof = HyperKZGProof<Bn254>;
    type OpeningProofHint = ();

    fn test_setup<R: ark_std::rand::Rng + ark_std::rand::RngCore>(
        rng: &mut R,
        max_num_vars: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        let srs = HyperKZGSRS::setup(rng, 1 << max_num_vars);
        let (cpu_pk, vk) = srs.trim(1 << max_num_vars);
        (HyperKZGGpuProverKey::from_cpu(&cpu_pk), vk)
    }

    #[tracing::instrument(skip_all, name = "HyperKZGGpu::commit")]
    fn commit(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)> {
        let results = gpu_batch_commit(setup.bases_gpu(), &[poly])?;
        Ok((HyperKZGCommitment(results[0].into_affine()), ()))
    }

    #[tracing::instrument(skip_all, name = "HyperKZGGpu::batch_commit")]
    fn batch_commit<'a, U>(
        gens: &Self::ProverSetup,
        polys: &[U],
    ) -> anyhow::Result<Vec<(Self::Commitment, Self::OpeningProofHint)>>
    where
        U: Borrow<DensePolynomial<'a, Self::Field>> + Sync,
    {
        let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().map(|p| p.borrow()).collect();
        let results = gpu_batch_commit(gens.bases_gpu(), &poly_refs)?;

        Ok(results
            .into_iter()
            .map(|r| (HyperKZGCommitment(r.into_affine()), ()))
            .collect())
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> anyhow::Result<Self::Commitment> {
        let points: Vec<G1Affine> = commitments.iter().map(|c| c.borrow().0).collect();
        let result = G1Projective::msm(&points, coeffs)
            .map_err(|e| anyhow::anyhow!("MSM failed with length mismatch: {e}"))?;
        Ok(HyperKZGCommitment(result.into_affine()))
    }

    #[tracing::instrument(skip_all, name = "HyperKZGGpu::prove")]
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<Self::Proof> {
        // open_gpu ignores the eval parameter, so skip the expensive poly.evaluate()
        Self::open_gpu(setup, poly, opening_point, &Fr::ZERO, transcript)
    }

    fn verify<ProofTranscript: Transcript>(
        setup: &Self::VerifierSetup,
        proof: &Self::Proof,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> anyhow::Result<()> {
        super::HyperKZG::<Bn254>::verify(
            setup,
            commitment,
            opening_point,
            opening,
            proof,
            transcript,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"HyperKZGGpu"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkyper::HyperKZG;
    use crate::arkyper::transcript::blake3::Blake3Transcript;
    use crate::poly::challenge;
    use ark_std::UniformRand;
    use ark_std::rand::SeedableRng;

    /// Test that GPU fix_var matches CPU fix_var.
    #[test]
    fn test_fix_var_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        for log_n in [4, 8, 10] {
            let n = 1 << log_n;
            let poly: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let r = Fr::rand(&mut rng);

            let cpu_result = cpu_fix_var(&poly, &r);
            let gpu_result = gpu_fix_var(&poly, &r).expect("GPU fix_var failed");

            assert_eq!(
                cpu_result.len(),
                gpu_result.len(),
                "Length mismatch for n={n}"
            );
            for (i, (cpu, gpu)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
                assert_eq!(cpu, gpu, "Mismatch at index {i} for n={n}");
            }
        }
    }

    /// Test that GPU fix_vars_with_intermediates matches CPU version.
    #[test]
    fn test_fix_vars_intermediates_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(43);

        for ell in [4, 6, 8] {
            let n = 1 << ell;
            let poly: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let challenges: Vec<Fr> = (0..ell - 1).map(|_| Fr::rand(&mut rng)).collect();

            let cpu_results = cpu_fix_vars_with_intermediates(&poly, &challenges);
            let gpu_results =
                gpu_fix_vars_with_intermediates(&poly, &challenges).expect("GPU failed");

            assert_eq!(cpu_results.len(), gpu_results.len());
            for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
                assert_eq!(cpu.len(), gpu.len(), "Length mismatch at step {i}");
                for (j, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                    assert_eq!(c, g, "Mismatch at step {i}, index {j}");
                }
            }
        }
    }

    /// Test that GPU batch_commit matches CPU batch_commit.
    #[test]
    fn test_batch_commit_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(44);

        for ell in [8, 10, 12] {
            let n = 1 << ell;
            let num_polys = 5;

            let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
                .map(|_| {
                    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                    DensePolynomial::new(coeffs)
                })
                .collect();

            let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
            let (cpu_pk, _) = srs.trim(n);
            let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

            let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();

            let cpu_results = cpu_batch_commit(cpu_pk.g1_powers(), &poly_refs).expect("CPU failed");
            let gpu_results = gpu_batch_commit(gpu_pk.bases_gpu(), &poly_refs).expect("GPU failed");

            assert_eq!(cpu_results.len(), gpu_results.len());
            for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
                assert_eq!(
                    cpu.into_affine(),
                    gpu.into_affine(),
                    "Commitment mismatch for poly {i}"
                );
            }
        }
    }

    /// Test that GPU open matches CPU open and produces valid proofs.
    #[test]
    fn test_open_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(45);

        for ell in [8, 10] {
            let n = 1 << ell;

            let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DensePolynomial::from(poly_raw);
            let point: Vec<Fr> = (0..ell)
                .map(|_| challenge::random_challenge::<Fr, _>(&mut rng))
                .collect();
            let eval = poly.evaluate(&point).expect("eval failed");

            let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
            let (cpu_pk, vk) = srs.trim(n);
            let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

            let (comm, _) = HyperKZG::<Bn254>::commit(&cpu_pk, &poly).expect("commit failed");

            // CPU open
            let mut cpu_transcript = Blake3Transcript::new(b"TestOpen");
            let cpu_proof =
                HyperKZGGpu::<Bn254>::open_cpu(&cpu_pk, &poly, &point, &eval, &mut cpu_transcript)
                    .expect("CPU open failed");

            // GPU open
            let mut gpu_transcript = Blake3Transcript::new(b"TestOpen");
            let gpu_proof =
                HyperKZGGpu::<Bn254>::open_gpu(&gpu_pk, &poly, &point, &eval, &mut gpu_transcript)
                    .expect("GPU open failed");

            // Compare intermediate commitments
            assert_eq!(
                cpu_proof.coms.len(),
                gpu_proof.coms.len(),
                "Number of commitments differs"
            );
            for (i, (cpu_com, gpu_com)) in
                cpu_proof.coms.iter().zip(gpu_proof.coms.iter()).enumerate()
            {
                assert_eq!(
                    cpu_com, gpu_com,
                    "Commitment {i} differs between CPU and GPU"
                );
            }

            // Compare witness commitments
            assert_eq!(cpu_proof.w.len(), gpu_proof.w.len(), "w length differs");
            for (i, (cpu_w, gpu_w)) in cpu_proof.w.iter().zip(gpu_proof.w.iter()).enumerate() {
                assert_eq!(cpu_w, gpu_w, "w[{i}] differs between CPU and GPU");
            }

            // Compare v values
            assert_eq!(cpu_proof.v.len(), gpu_proof.v.len(), "v length differs");
            for (i, (cpu_v, gpu_v)) in cpu_proof.v.iter().zip(gpu_proof.v.iter()).enumerate() {
                assert_eq!(cpu_v, gpu_v, "v[{i}] differs between CPU and GPU");
            }

            // Verify both proofs
            let mut verify_transcript = Blake3Transcript::new(b"TestOpen");
            HyperKZG::<Bn254>::verify(
                &vk,
                &comm,
                &point,
                &eval,
                &cpu_proof,
                &mut verify_transcript,
            )
            .expect("CPU proof verification failed");

            let mut verify_transcript = Blake3Transcript::new(b"TestOpen");
            HyperKZG::<Bn254>::verify(
                &vk,
                &comm,
                &point,
                &eval,
                &gpu_proof,
                &mut verify_transcript,
            )
            .expect("GPU proof verification failed");
        }
    }

    /// Full integration test: HyperKZGGpu trait implementation vs HyperKZG.
    #[test]
    fn test_hyperkzg_gpu_trait_vs_cpu_trait() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(46);

        for ell in [8, 10] {
            let n = 1 << ell;

            let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DensePolynomial::from(poly_raw);
            let point: Vec<Fr> = (0..ell)
                .map(|_| challenge::random_challenge::<Fr, _>(&mut rng))
                .collect();
            let eval = poly.evaluate(&point).expect("eval failed");

            // Generate both CPU and GPU prover keys from the same SRS
            let srs = HyperKZGSRS::setup(&mut rng, n);
            let (cpu_pk, vk) = srs.trim(n);
            let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

            // CPU commit (using original HyperKZG)
            let (cpu_comm, _) =
                HyperKZG::<Bn254>::commit(&cpu_pk, &poly).expect("CPU commit failed");

            // GPU commit (using HyperKZGGpu)
            let (gpu_comm, _) =
                HyperKZGGpu::<Bn254>::commit(&gpu_pk, &poly).expect("GPU commit failed");

            assert_eq!(
                cpu_comm.0, gpu_comm.0,
                "Commitments differ between CPU and GPU"
            );

            // CPU prove (using original HyperKZG)
            let mut cpu_transcript = Blake3Transcript::new(b"TraitTest");
            let cpu_proof =
                HyperKZG::<Bn254>::prove(&cpu_pk, &poly, &point, None, &mut cpu_transcript)
                    .expect("CPU prove failed");

            // GPU prove (using HyperKZGGpu trait)
            let mut gpu_transcript = Blake3Transcript::new(b"TraitTest");
            let gpu_proof =
                HyperKZGGpu::<Bn254>::prove(&gpu_pk, &poly, &point, None, &mut gpu_transcript)
                    .expect("GPU prove failed");

            // Verify both proofs
            let mut verify_transcript = Blake3Transcript::new(b"TraitTest");
            HyperKZG::<Bn254>::verify(
                &vk,
                &cpu_comm,
                &point,
                &eval,
                &cpu_proof,
                &mut verify_transcript,
            )
            .expect("CPU proof verification failed");

            let mut verify_transcript = Blake3Transcript::new(b"TraitTest");
            HyperKZGGpu::<Bn254>::verify(
                &vk,
                &gpu_proof,
                &mut verify_transcript,
                &point,
                &eval,
                &gpu_comm,
            )
            .expect("GPU proof verification failed");

            println!("ell={ell}: CPU and GPU produce matching results");
        }
    }

    /// Load g1_powers from a SRS file written by test_generate_srs,
    /// then convert to GPU prover key.
    fn load_gpu_pk_from_file(path: &str) -> HyperKZGGpuProverKey {
        use ark_serialize::CanonicalDeserialize;
        use std::fs::File;
        use std::io::BufReader;

        let mut reader = BufReader::new(File::open(path).unwrap_or_else(|e| {
            panic!("SRS file not found at {path}: {e}. Run test_generate_srs first.")
        }));
        let powers: Vec<G1Affine> = CanonicalDeserialize::deserialize_compressed(&mut reader)
            .expect("deserialize g1_powers failed");
        let cpu_pk = HyperKZGProverKey {
            kzg_pk: ark_poly_commit::kzg10::Powers {
                powers_of_g: std::borrow::Cow::Owned(powers),
                powers_of_gamma_g: std::borrow::Cow::Owned(vec![]),
            },
        };
        HyperKZGGpuProverKey::from_cpu(&cpu_pk)
    }

    /// GPU batch_commit measurement from exported data.
    /// Loads pre-generated SRS from disk and converts to GPU format.
    #[test]
    fn test_gpu_commit_from_exported_data() {
        use crate::arkyper::PcsCommitExport;
        use std::fs::File;
        use std::io::BufReader;
        use std::time::Instant;

        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        println!("[gpu-commit] Loading commit polys...");
        let t0 = Instant::now();
        let (polys, num_polys, max_len) = {
            let file = match File::open("/tmp/pcs_commit_polys.bin") {
                Ok(f) => f,
                Err(_) => {
                    println!("No export file found, skipping");
                    return;
                }
            };
            let export: PcsCommitExport<Fr> =
                PcsCommitExport::read_canonical(&mut BufReader::new(file))
                    .expect("deserialize failed");
            let polys: Vec<DensePolynomial<Fr>> = export
                .polys
                .into_iter()
                .map(|e| DensePolynomial::new(e.evals))
                .collect();
            let num_polys = polys.len();
            let max_len = polys.iter().map(|p| p.len()).max().unwrap();
            (polys, num_polys, max_len)
        };
        println!(
            "[gpu-commit] Loaded {} polys in {:.2?} - max_len = {}, sizes = {:?}",
            num_polys,
            t0.elapsed(),
            max_len,
            polys.iter().map(|p| p.num_vars()).collect::<Vec<_>>()
        );

        let srs_path = format!("/tmp/pcs_srs_{}.bin", max_len);
        println!("[gpu-commit] Loading SRS from {}...", srs_path);
        let t_load = Instant::now();
        let gpu_pk = load_gpu_pk_from_file(&srs_path);
        let load_time = t_load.elapsed();
        println!("[gpu-commit] SRS loaded + converted: {:.2?}", load_time);

        let t_commit = Instant::now();
        let _commits =
            HyperKZGGpu::<Bn254>::batch_commit(&gpu_pk, &polys).expect("GPU batch_commit failed");
        let commit_time = t_commit.elapsed();

        println!(
            "=== GPU commit: load {:.2?}, batch_commit {:.2?} ({} polys) ===",
            load_time, commit_time, num_polys
        );
    }

    /// GPU prove measurement from exported data.
    /// Loads pre-generated SRS from disk and converts to GPU format.
    #[test]
    fn test_gpu_open_from_exported_data() {
        use crate::arkyper::PcsOpenExport;
        use std::fs::File;
        use std::io::BufReader;
        use std::time::Instant;

        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        println!("[gpu-open] Loading open poly...");
        let (poly, point) = {
            let file = match File::open("/tmp/pcs_open_poly.bin") {
                Ok(f) => f,
                Err(_) => {
                    println!("No export file found, skipping");
                    return;
                }
            };
            let export: PcsOpenExport<Fr> =
                PcsOpenExport::read_canonical(&mut BufReader::new(file))
                    .expect("deserialize failed");
            (DensePolynomial::new(export.poly.evals), export.point)
        };
        let max_len = poly.len();
        println!("[gpu-open] Loaded poly nv={}", poly.num_vars());

        let srs_path = format!("/tmp/pcs_srs_{}.bin", max_len);
        println!("[gpu-open] Loading SRS from {}...", srs_path);
        let t_load = Instant::now();
        let gpu_pk = load_gpu_pk_from_file(&srs_path);
        let load_time = t_load.elapsed();
        println!("[gpu-open] SRS loaded + converted: {:.2?}", load_time);

        let t_prove = Instant::now();
        let mut transcript = Blake3Transcript::new(b"ExportedTest");
        let _proof = HyperKZGGpu::<Bn254>::prove(&gpu_pk, &poly, &point, None, &mut transcript)
            .expect("GPU prove failed");
        let prove_time = t_prove.elapsed();

        println!(
            "=== GPU open: load {:.2?}, prove {:.2?} ===",
            load_time, prove_time
        );
    }

    /// Batch commit comparison test.
    #[test]
    fn test_batch_commit_trait_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(47);
        let ell = 10;
        let n = 1 << ell;
        let num_polys = 10;

        let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs)
            })
            .collect();

        // Generate both CPU and GPU prover keys from the same SRS
        let srs = HyperKZGSRS::setup(&mut rng, n);
        let (cpu_pk, _) = srs.trim(n);
        let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

        // CPU batch commit
        let cpu_commits = HyperKZG::<Bn254>::batch_commit(&cpu_pk, &polys).expect("CPU failed");

        // GPU batch commit
        let gpu_commits = HyperKZGGpu::<Bn254>::batch_commit(&gpu_pk, &polys).expect("GPU failed");

        assert_eq!(cpu_commits.len(), gpu_commits.len());
        for (i, ((cpu_c, _), (gpu_c, _))) in cpu_commits.iter().zip(gpu_commits.iter()).enumerate()
        {
            assert_eq!(
                cpu_c.0, gpu_c.0,
                "Batch commit {i} differs between CPU and GPU"
            );
        }

        println!(
            "Batch commit test passed: {} polynomials, n={}",
            num_polys, n
        );
    }

    /// Test GPU SRS from_cpu conversion produces equivalent commits.
    #[test]
    fn test_gpu_srs_from_cpu_equivalence() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(48);
        let ell = 10;
        let n = 1 << ell;

        let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DensePolynomial::from(poly_raw);

        // CPU SRS → CPU commit
        let cpu_srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (cpu_pk, vk) = cpu_srs.trim(n);
        let (cpu_comm, _) = HyperKZG::<Bn254>::commit(&cpu_pk, &poly).expect("CPU commit failed");

        // CPU SRS → GPU SRS via from_cpu → GPU commit
        let cpu_srs2 = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let gpu_srs = HyperKZGGpuSRS::from_cpu(cpu_srs2);
        let (gpu_pk, gpu_vk) = gpu_srs.trim(n);
        let (gpu_comm, _) =
            HyperKZGGpu::<Bn254>::commit(&gpu_pk, &poly).expect("GPU commit failed");

        // Note: different SRS → different commits (different random secrets).
        // But both should produce valid commitments that verify.
        // To compare same-SRS results, use from_cpu on the prover key:
        let gpu_pk_same = HyperKZGGpuProverKey::from_cpu(&cpu_pk);
        let (gpu_comm_same, _) =
            HyperKZGGpu::<Bn254>::commit(&gpu_pk_same, &poly).expect("GPU commit same SRS failed");

        assert_eq!(
            cpu_comm.0, gpu_comm_same.0,
            "CPU and GPU (from_cpu) commits should match for same SRS"
        );
        println!("GPU SRS from_cpu equivalence test passed");
    }

    /// Test serialization roundtrip for HyperKZGGpuProverKey.
    #[test]
    fn test_gpu_prover_key_serialization() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(49);
        let n = 1 << 8;

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (cpu_pk, _) = srs.trim(n);
        let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

        // Serialize
        let serialized = rmp_serde::to_vec(&gpu_pk).expect("serialize failed");
        println!(
            "Serialized GPU prover key: {} bytes ({} bases)",
            serialized.len(),
            gpu_pk.bases_gpu().len()
        );

        // Deserialize
        let deserialized: HyperKZGGpuProverKey =
            rmp_serde::from_slice(&serialized).expect("deserialize failed");

        // Compare
        assert_eq!(
            gpu_pk.bases_gpu().len(),
            deserialized.bases_gpu().len(),
            "Length mismatch after roundtrip"
        );
        for (i, (orig, deser)) in gpu_pk
            .bases_gpu()
            .iter()
            .zip(deserialized.bases_gpu().iter())
            .enumerate()
        {
            assert_eq!(orig.x, deser.x, "x mismatch at index {i}");
            assert_eq!(orig.y, deser.y, "y mismatch at index {i}");
        }

        // Verify commits match
        let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DensePolynomial::from(poly_raw);

        let (orig_comm, _) =
            HyperKZGGpu::<Bn254>::commit(&gpu_pk, &poly).expect("commit with original failed");
        let (deser_comm, _) = HyperKZGGpu::<Bn254>::commit(&deserialized, &poly)
            .expect("commit with deserialized failed");

        assert_eq!(
            orig_comm.0, deser_comm.0,
            "Commits differ after serialization roundtrip"
        );
        println!("GPU prover key serialization roundtrip test passed");
    }

    /// Test that GPU batch_commit with mixed-size polynomials matches CPU.
    ///
    /// Creates polys of varying sizes (2^7 through 2^14) to exercise the
    /// size-grouping logic: tiny polys use CPU MSM, larger ones use GPU.
    #[test]
    fn test_batch_commit_mixed_sizes() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(50);

        // Mixed sizes: some below GPU_MSM_THRESHOLD (4096), some above
        let sizes = [1 << 7, 1 << 10, 1 << 12, 1 << 14];
        let max_size = *sizes.iter().max().unwrap();

        let mut polys = Vec::new();
        for &n in &sizes {
            for _ in 0..3 {
                let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                polys.push(DensePolynomial::new(coeffs));
            }
        }

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, max_size);
        let (cpu_pk, _) = srs.trim(max_size);
        let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

        let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();

        // CPU: commit each poly individually with correct base slice
        let cpu_results = cpu_batch_commit(cpu_pk.g1_powers(), &poly_refs).expect("CPU failed");

        // GPU: batch commit with mixed sizes (exercises grouping)
        let gpu_results = gpu_batch_commit(gpu_pk.bases_gpu(), &poly_refs).expect("GPU failed");

        assert_eq!(cpu_results.len(), gpu_results.len());
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert_eq!(
                cpu.into_affine(),
                gpu.into_affine(),
                "Commitment mismatch for poly {i} (size {})",
                polys[i].len()
            );
        }

        println!(
            "Mixed-size batch commit test passed: {} polys, sizes {:?}",
            polys.len(),
            sizes
        );
    }

    /// Test that GPU batch_commit (Pippenger) produces identical results to CPU MSM.
    ///
    /// Commits 128 same-size polys via GPU batch_commit and CPU MSM, asserting
    /// they produce identical commitments.
    #[test]
    fn test_batch_commit_pippenger_correctness() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(55);

        let ell = 13; // 2^13 = 8192 elements per poly
        let n = 1 << ell;
        let num_polys = 128;

        let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs)
            })
            .collect();

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (cpu_pk, _) = srs.trim(n);
        let gpu_pk = HyperKZGGpuProverKey::from_cpu(&cpu_pk);

        let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();

        // CPU reference
        let cpu_results = cpu_batch_commit(cpu_pk.g1_powers(), &poly_refs)
            .expect("CPU batch_commit failed");

        // GPU Pippenger
        let gpu_results = gpu_batch_commit(gpu_pk.bases_gpu(), &poly_refs)
            .expect("GPU batch_commit failed");

        assert_eq!(cpu_results.len(), gpu_results.len());
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert_eq!(
                cpu.into_affine(),
                gpu.into_affine(),
                "Commitment mismatch at poly {i}: CPU vs GPU"
            );
        }

        println!(
            "batch_commit Pippenger correctness test passed: {} polys of size {}",
            num_polys, n
        );
    }
}
