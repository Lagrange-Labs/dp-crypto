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

use std::borrow::Borrow;
use std::marker::PhantomData;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::AdditiveGroup;
use std::ops::Mul;

use super::gpu_msm::convert_bases_to_gpu;
use super::transcript::Transcript;
use super::{
    kzg_open_batch, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
    HyperKZGSRS,
};
use crate::arkyper::interface::CommitmentScheme;
use crate::poly::dense::DensePolynomial;
use ec_gpu::arkworks_bn254::G1Affine as GpuG1Affine;
use ec_gpu_gen::{
    compute_work_units, program, rust_gpu_tools::Device, FusedPolyCommit, G1AffineM, Phase3Input,
    PolyOpsKernel,
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

/// Batch commit using GPU - single call for multiple polynomials.
///
/// Uses the sort-based MSM pipeline via `FusedPolyCommit::batch_commit`,
/// which is significantly faster than the old per-thread bucket approach.
/// Bases are uploaded once and Montgomery→standard conversion happens on GPU.
pub fn gpu_batch_commit(
    g1_powers: &[G1Affine],
    polys: &[&DensePolynomial<Fr>],
) -> anyhow::Result<Vec<G1Projective>> {
    let _span = tracing::debug_span!("gpu_batch_commit", n_polys = polys.len()).entered();
    if polys.is_empty() {
        return Ok(vec![]);
    }

    let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);

    // Pass original poly slices directly — batch_commit handles zero-padding
    // internally per-poly, avoiding bulk allocation of all padded polys at once.
    let poly_slices: Vec<&[Fr]> = polys.iter().map(|p| p.evals_ref()).collect();

    let mut guard = GPU_FUSED.lock().unwrap();
    guard.ensure_bases_uploaded(&g1_powers[..max_len])?;
    let fused = guard.fused.as_ref().unwrap();
    let bases_gpu = &guard.cached_bases_gpu.as_ref().unwrap().1;
    fused
        .batch_commit(&poly_slices, bases_gpu)
        .map_err(|e| anyhow::anyhow!("GPU batch commit error: {e}"))
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
    cached_bases_gpu: Option<(usize, Vec<G1AffineM>)>,
}

impl GpuFusedHolder {
    pub fn new() -> Self {
        Self {
            fused: None,
            cached_bases_gpu: None,
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

            let fused = FusedPolyCommit::create(prog, wu)
                .map_err(|e| anyhow::anyhow!("Failed to create FusedPolyCommit: {e}"))?;

            self.fused = Some(fused);
        }
        Ok(())
    }

    /// Ensure SRS bases are uploaded to GPU as a persistent buffer.
    ///
    /// Converts bases to GPU format (cached), then uploads to GPU memory once.
    /// Subsequent calls with the same-sized bases are no-ops.
    pub fn ensure_bases_uploaded(&mut self, bases: &[G1Affine]) -> anyhow::Result<()> {
        self.ensure_init()?;
        // Ensure CPU-side cache is populated (side effect: populates self.cached_bases_gpu)
        let _ = self.get_or_convert_bases(bases);
        // Upload to GPU persistent buffer using split borrows
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
/// Computes `[G, tau*G, tau^2*G, ..., tau^n*G]` and `[gamma*G, gamma*tau*G, ...]`
/// using GPU batch scalar multiplication, which is significantly faster than
/// the CPU sequential approach for large degrees.
///
/// # Arguments
/// * `rng` - Random number generator for generating tau, gamma, and base points
/// * `max_degree` - Maximum polynomial degree (number of G1 powers = max_degree + 1)
///
/// # Returns
/// `HyperKZGSRS<Bn254>` containing the computed powers.
pub fn gpu_setup<R: ark_std::rand::Rng + ark_std::rand::RngCore>(
    rng: &mut R,
    max_degree: usize,
) -> anyhow::Result<super::HyperKZGSRS<Bn254>> {
    use ark_ec::CurveGroup;
    use ark_ff::{One, UniformRand};
    use ark_poly_commit::kzg10::UniversalParams;
    use std::collections::BTreeMap;

    if max_degree < 1 {
        return Err(anyhow::anyhow!("Degree must be at least 1"));
    }

    let _span = tracing::debug_span!("gpu_setup", max_degree).entered();

    // Generate random secrets (same as arkworks KZG10::setup)
    let beta = Fr::rand(rng); // tau in our notation
    let g = G1Projective::rand(rng);
    let gamma_g = G1Projective::rand(rng);
    let h = ark_bn254::G2Projective::rand(rng);

    // Compute powers of beta: [1, beta, beta^2, ..., beta^(max_degree+1)]
    let powers_of_beta_span = tracing::debug_span!("compute_powers_of_beta").entered();
    let mut powers_of_beta = vec![Fr::one()];
    let mut cur = beta;
    for _ in 0..=max_degree {
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
        .batch_scalar_mul(&g_gpu, &powers_of_beta[0..max_degree + 1])
        .map_err(|e| anyhow::anyhow!("GPU batch_scalar_mul error: {e}"))?;
    drop(powers_of_g_span);

    // GPU batch scalar multiplication for powers_of_gamma_g
    let gamma_g_affine: G1Affine = gamma_g.into_affine();
    let gamma_g_gpu: GpuG1Affine = GpuG1Affine::from(gamma_g_affine);

    let powers_of_gamma_g_span =
        tracing::debug_span!("gpu_batch_scalar_mul_powers_of_gamma_g").entered();
    let powers_of_gamma_g: BTreeMap<usize, G1Affine> = GPU_FUSED
        .lock()
        .unwrap()
        .get_or_init()?
        .batch_scalar_mul(&gamma_g_gpu, &powers_of_beta)
        .map_err(|e| anyhow::anyhow!("GPU batch_scalar_mul error: {e}"))?
        .into_iter()
        .enumerate()
        .collect();
    drop(powers_of_gamma_g_span);

    // G2 computations (small, done on CPU)
    let h_affine = h.into_affine();
    let beta_h = h.into_affine().mul(beta).into_affine();
    let prepared_h = h_affine.into();
    let prepared_beta_h = beta_h.into();

    let params = UniversalParams {
        powers_of_g,
        powers_of_gamma_g,
        h: h_affine,
        beta_h,
        neg_powers_of_h: BTreeMap::new(), // Not used in HyperKZG
        prepared_h,
        prepared_beta_h,
    };

    Ok(super::HyperKZGSRS::from_params(params))
}

impl HyperKZGGpu<Bn254> {
    /// GPU-accelerated open operation using fused GPU session.
    ///
    /// Runs the entire HyperKZG open in a single GPU session:
    /// - Bases uploaded once and reused for Phase 2 (intermediate commits) and Phase 3 (witness commits)
    /// - Scalar conversion happens on GPU (no CPU `convert_scalars_to_bigint`)
    /// - Witness polynomials never leave GPU
    /// - CPU transcript work runs inside the GPU session via callback
    pub fn open_gpu<ProofTranscript: Transcript>(
        pk: &HyperKZGProverKey<Bn254>,
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

        // Single lock scope: ensure bases are on GPU (persistent), then run fused_open
        let result = {
            let mut guard = GPU_FUSED.lock().unwrap();
            let upload_start = std::time::Instant::now();
            guard.ensure_bases_uploaded(pk.g1_powers())?;
            eprintln!("[open_gpu] ensure_bases_uploaded: {:?}", upload_start.elapsed());
            // After ensure_bases_uploaded, the persistent buffer is active on the FusedPolyCommit.
            // fused_open will use the persistent buffer and skip re-upload.
            // The bases slice is still passed for assertion checks and fallback.
            let fused = guard.fused.as_ref().unwrap();
            let bases_gpu = &guard.cached_bases_gpu.as_ref().unwrap().1;
            fused
                .fused_open(
                    poly.evals_ref(),
                    challenges,
                    bases_gpu,
                    pk.g1_powers(),
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

                        eprintln!("[open_gpu]   callback transcript: {:?}", callback_detail_start.elapsed());
                        let eval_start = std::time::Instant::now();

                        // Evaluate f_i(u_j) on CPU using Horner's method on raw slices,
                        // parallelized with rayon for large polynomial counts.
                        let k = poly_slices.len();
                        let t = u.len();
                        let mut v = vec![vec![Fr::ZERO; k]; t];
                        {
                            use rayon::prelude::*;
                            v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
                                v_i.par_iter_mut()
                                    .zip(poly_slices.par_iter())
                                    .for_each(|(v_ij, coeffs)| {
                                        *v_ij = eval_as_univariate(coeffs, &u[i]);
                                    });
                            });
                        }

                        eprintln!("[open_gpu]   callback evals ({}x{}): {:?}", t, k, eval_start.elapsed());

                        // Transcript: append evals and get q_powers
                        let scalars: Vec<&Fr> = v.iter().flatten().collect();
                        transcript.append_scalars::<Fr>(&scalars);
                        let q_powers: Vec<Fr> = transcript.challenge_scalar_powers(k);

                        // No padded_polys needed — intermediate buffers are kept on GPU
                        // and packed via copy_and_pad kernel, eliminating ~2.8GB CPU→GPU transfer.
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

        eprintln!("[open_gpu] fused_open returned: {:?}", open_gpu_start.elapsed());

        // Final transcript work (witness commitments)
        let final_transcript_start = std::time::Instant::now();
        let w_aff: Vec<G1Affine> = result
            .witness_commitments
            .iter()
            .map(|g| g.into_affine())
            .collect();
        transcript.append_points(&w_aff);
        let _d_0: Fr = transcript.challenge_scalar();
        eprintln!("[open_gpu] final transcript: {:?}", final_transcript_start.elapsed());
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
    type ProverSetup = HyperKZGProverKey<Bn254>;
    type VerifierSetup = HyperKZGVerifierKey<Bn254>;
    type Commitment = HyperKZGCommitment<Bn254>;
    type Proof = HyperKZGProof<Bn254>;
    type BatchedProof = HyperKZGProof<Bn254>;
    type OpeningProofHint = ();

    fn test_setup<R: ark_std::rand::Rng + ark_std::rand::RngCore>(
        rng: &mut R,
        max_num_vars: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        HyperKZGSRS::setup(rng, 1 << max_num_vars).trim(1 << max_num_vars)
    }

    #[tracing::instrument(skip_all, name = "HyperKZGGpu::commit")]
    fn commit(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)> {
        let results = gpu_batch_commit(setup.g1_powers(), &[poly])?;
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
        let results = gpu_batch_commit(gens.g1_powers(), &poly_refs)?;

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
        // Use the existing verify implementation
        super::HyperKZG::<Bn254>::verify(setup, commitment, opening_point, opening, proof, transcript)
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
    use crate::arkyper::transcript::blake3::Blake3Transcript;
    use crate::arkyper::HyperKZG;
    use crate::poly::challenge;
    use ark_std::rand::SeedableRng;
    use ark_std::UniformRand;

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

            // Generate random polynomials
            let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
                .map(|_| {
                    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                    DensePolynomial::new(coeffs)
                })
                .collect();

            // Setup SRS
            let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
            let (pk, _) = srs.trim(n);

            let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();

            let cpu_results = cpu_batch_commit(pk.g1_powers(), &poly_refs).expect("CPU failed");
            let gpu_results = gpu_batch_commit(pk.g1_powers(), &poly_refs).expect("GPU failed");

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
            let (pk, vk) = srs.trim(n);

            // Get commitment (should be same for both)
            let (comm, _) = HyperKZG::<Bn254>::commit(&pk, &poly).expect("commit failed");

            // CPU open
            let mut cpu_transcript = Blake3Transcript::new(b"TestOpen");
            let cpu_proof =
                HyperKZGGpu::<Bn254>::open_cpu(&pk, &poly, &point, &eval, &mut cpu_transcript)
                    .expect("CPU open failed");

            // GPU open
            let mut gpu_transcript = Blake3Transcript::new(b"TestOpen");
            let gpu_proof =
                HyperKZGGpu::<Bn254>::open_gpu(&pk, &poly, &point, &eval, &mut gpu_transcript)
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
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &cpu_proof, &mut verify_transcript)
                .expect("CPU proof verification failed");

            let mut verify_transcript = Blake3Transcript::new(b"TestOpen");
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &gpu_proof, &mut verify_transcript)
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

            // Setup using trait
            let (pk, vk) = HyperKZGGpu::<Bn254>::test_setup(&mut rng, ell);

            // CPU commit (using original HyperKZG)
            let (cpu_comm, _) =
                HyperKZG::<Bn254>::commit(&pk, &poly).expect("CPU commit failed");

            // GPU commit (using HyperKZGGpu)
            let (gpu_comm, _) =
                HyperKZGGpu::<Bn254>::commit(&pk, &poly).expect("GPU commit failed");

            assert_eq!(
                cpu_comm.0, gpu_comm.0,
                "Commitments differ between CPU and GPU"
            );

            // CPU prove (using original HyperKZG)
            let mut cpu_transcript = Blake3Transcript::new(b"TraitTest");
            let cpu_proof = HyperKZG::<Bn254>::prove(&pk, &poly, &point, None, &mut cpu_transcript)
                .expect("CPU prove failed");

            // GPU prove (using HyperKZGGpu trait)
            let mut gpu_transcript = Blake3Transcript::new(b"TraitTest");
            let gpu_proof =
                HyperKZGGpu::<Bn254>::prove(&pk, &poly, &point, None, &mut gpu_transcript)
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

        let (pk, _) = HyperKZGGpu::<Bn254>::test_setup(&mut rng, ell);

        // CPU batch commit
        let cpu_commits = HyperKZG::<Bn254>::batch_commit(&pk, &polys).expect("CPU failed");

        // GPU batch commit
        let gpu_commits = HyperKZGGpu::<Bn254>::batch_commit(&pk, &polys).expect("GPU failed");

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
}
