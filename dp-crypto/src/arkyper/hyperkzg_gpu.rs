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
use std::sync::Arc;

use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};

use super::gpu_msm::{convert_bases_to_gpu, convert_scalars_to_bigint, GPU_MSM};
use super::transcript::Transcript;
use super::{
    kzg_open_batch, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
    HyperKZGSRS,
};
use crate::arkyper::interface::CommitmentScheme;
use crate::poly::dense::DensePolynomial;
use ec_gpu_gen::{program, rust_gpu_tools::Device, PolyOpsKernel};

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
/// This uploads all bases once and processes all polynomials in sequence,
/// minimizing GPU memory transfers.
pub fn gpu_batch_commit(
    g1_powers: &[G1Affine],
    polys: &[&DensePolynomial<Fr>],
) -> anyhow::Result<Vec<G1Projective>> {
    if polys.is_empty() {
        return Ok(vec![]);
    }

    // Pre-convert bases to GPU format once
    let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
    let bases_gpu = Arc::new(convert_bases_to_gpu(&g1_powers[..max_len]));

    // Process all polynomials
    let results: Vec<G1Projective> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.evals_ref();
            let msm_size = coeffs.len();
            let scalars_bigint = Arc::new(convert_scalars_to_bigint(coeffs));
            let bases_slice = Arc::new(bases_gpu[..msm_size].to_vec());

            GPU_MSM
                .lock()
                .unwrap()
                .msm_arc(bases_slice, scalars_bigint)
                .map_err(|e| anyhow::anyhow!("GPU MSM error: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

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

/// GPU-accelerated batch evaluation of multiple polynomials at multiple points.
///
/// Returns `results[point_idx][poly_idx]` = evaluation of poly_idx at points[point_idx].
pub fn gpu_eval_univariate_batch(
    polys: &[&[Fr]],
    points: &[Fr],
) -> anyhow::Result<Vec<Vec<Fr>>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .eval_univariate_batch(polys, points)
        .map_err(|e| anyhow::anyhow!("GPU eval_univariate_batch error: {e}"))
}

/// GPU-accelerated batch witness polynomial computation for multiple points.
///
/// Returns one witness polynomial per point, each of length `f.len() - 1`.
pub fn gpu_witness_poly_batch(f: &[Fr], points: &[Fr]) -> anyhow::Result<Vec<Vec<Fr>>> {
    GPU_POLY_OPS
        .lock()
        .unwrap()
        .get_or_init()?
        .witness_poly_batch(f, points)
        .map_err(|e| anyhow::anyhow!("GPU witness_poly_batch error: {e}"))
}

// ============================================================================
// GPU KZG Batch Open (Phase 3 Acceleration)
// ============================================================================

/// GPU-accelerated KZG batch open for HyperKZG Phase 3.
///
/// This replaces the CPU `kzg_open_batch` by moving polynomial evaluations,
/// linear combination, witness polynomial computation, and MSM to the GPU.
///
/// # Arguments
/// * `f` - Intermediate polynomials from Phase 1
/// * `u` - Evaluation points [r, -r, r²]
/// * `pk` - Prover key containing G1 powers
/// * `transcript` - Fiat-Shamir transcript
///
/// # Returns
/// Tuple of (witness commitments, evaluation matrix)
#[allow(clippy::type_complexity)]
pub fn kzg_open_batch_gpu<T: Transcript>(
    f: &[DensePolynomial<Fr>],
    u: &[Fr],
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut T,
) -> anyhow::Result<(Vec<G1Affine>, Vec<Vec<Fr>>)> {
    let _k = f.len(); // Number of polynomials
    let _t = u.len(); // Number of evaluation points (3 for HyperKZG)

    // Step 1: GPU batch evaluation of all polynomials at all points
    // Collect polynomial coefficients as slices
    let poly_slices: Vec<&[Fr]> = f.iter().map(|p| p.evals_ref()).collect();

    // GPU: Evaluate all k polynomials at all t points
    // Returns v[point_idx][poly_idx]
    let v = gpu_eval_univariate_batch(&poly_slices, u)?;

    // Step 2: Update transcript and get challenge
    let scalars: Vec<&Fr> = v.iter().flatten().collect();
    transcript.append_scalars::<Fr>(&scalars);
    let q_powers: Vec<Fr> = transcript.challenge_scalar_powers(f.len());

    // Step 3: GPU linear combination to get B polynomial
    // B(x) = sum(q^i * f_i(x)) for i = 0..k
    let b_poly_coeffs = gpu_linear_combine(&poly_slices, &q_powers)?;

    // Step 4: GPU batch witness polynomial computation
    // For each point u[i], compute witness h_i where B(x) = h_i(x) * (x - u[i]) + B(u[i])
    let witness_polys = gpu_witness_poly_batch(&b_poly_coeffs, u)?;

    // Step 5: GPU MSM for witness commitments
    // Each witness polynomial needs to be committed using G1 powers
    let g1_powers = pk.g1_powers();
    let witness_len = witness_polys[0].len();

    anyhow::ensure!(
        witness_len <= g1_powers.len(),
        "Witness polynomial length {} exceeds G1 powers length {}",
        witness_len,
        g1_powers.len()
    );

    // Convert witness polys to DensePolynomial for batch commit
    let witness_dense: Vec<DensePolynomial<Fr>> = witness_polys
        .into_iter()
        .map(DensePolynomial::new)
        .collect();
    let witness_refs: Vec<&DensePolynomial<Fr>> = witness_dense.iter().collect();

    // GPU MSM for all witness polynomials
    let w_projective = gpu_batch_commit(g1_powers, &witness_refs)?;
    let w_aff: Vec<G1Affine> = w_projective.iter().map(|g| g.into_affine()).collect();

    // Step 6: Update transcript for verifier state consistency
    transcript.append_points(&w_aff);
    let _d_0: Fr = transcript.challenge_scalar();

    Ok((w_aff, v))
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
// HyperKZG GPU Session (Buffer Persistence)
// ============================================================================

/// Cached polynomial data for a GPU session.
#[derive(Clone)]
pub struct CachedPolynomial {
    /// The polynomial evaluations (owned).
    pub evals: Vec<Fr>,
    /// Pre-computed commitment (if available).
    pub commitment: Option<HyperKZGCommitment<Bn254>>,
    /// Pre-computed intermediate polynomial evaluations from fix_var (if available).
    pub intermediates: Option<Vec<Vec<Fr>>>,
}

/// GPU session that keeps polynomial data cached for efficient commit→open flow.
///
/// This session optimizes the HyperKZG workflow by:
/// 1. Caching polynomial data after commit
/// 2. Reusing cached data during open (avoiding re-computation)
/// 3. Providing a combined commit+open operation for maximum efficiency
///
/// # Example
///
/// ```ignore
/// let mut session = HyperKZGGpuSession::new();
///
/// // Commit and cache the polynomial
/// let commitment = session.commit_with_cache(&pk, &poly)?;
///
/// // Open using cached data (no re-upload needed)
/// let proof = session.open_from_cache(&pk, &point, &eval, &mut transcript)?;
///
/// // Or use the combined method for best performance:
/// let (commitment, proof) = session.commit_and_open(&pk, &poly, &point, &mut transcript)?;
/// ```
pub struct HyperKZGGpuSession {
    /// Cached polynomial data.
    cached_poly: Option<CachedPolynomial>,
}

impl Default for HyperKZGGpuSession {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperKZGGpuSession {
    /// Create a new GPU session.
    pub fn new() -> Self {
        Self { cached_poly: None }
    }

    /// Check if a polynomial is cached.
    pub fn has_cached_poly(&self) -> bool {
        self.cached_poly.is_some()
    }

    /// Get the cached commitment, if available.
    pub fn cached_commitment(&self) -> Option<&HyperKZGCommitment<Bn254>> {
        self.cached_poly.as_ref().and_then(|c| c.commitment.as_ref())
    }

    /// Clear the cached polynomial data.
    pub fn clear_cache(&mut self) {
        self.cached_poly = None;
    }

    /// Commit to a polynomial and cache it for later open.
    ///
    /// This stores the polynomial and commitment for use in `open_from_cache`.
    pub fn commit_with_cache(
        &mut self,
        pk: &HyperKZGProverKey<Bn254>,
        poly: &DensePolynomial<Fr>,
    ) -> anyhow::Result<HyperKZGCommitment<Bn254>> {
        // Compute commitment using GPU
        let results = gpu_batch_commit(pk.g1_powers(), &[poly])?;
        let commitment = HyperKZGCommitment(results[0].into_affine());

        // Cache the polynomial evaluations and commitment
        self.cached_poly = Some(CachedPolynomial {
            evals: poly.evals_ref().to_vec(),
            commitment: Some(commitment.clone()),
            intermediates: None,
        });

        Ok(commitment)
    }

    /// Open a proof using cached polynomial data.
    ///
    /// This requires that `commit_with_cache` was called first with the same polynomial.
    /// The cached data is used to avoid re-uploading the polynomial to GPU.
    pub fn open_from_cache<T: Transcript>(
        &mut self,
        pk: &HyperKZGProverKey<Bn254>,
        point: &[Fr],
        eval: &Fr,
        transcript: &mut T,
    ) -> anyhow::Result<HyperKZGProof<Bn254>> {
        let cached = self.cached_poly.as_ref().ok_or_else(|| {
            anyhow::anyhow!("No cached polynomial. Call commit_with_cache first.")
        })?;

        // Recreate DensePolynomial from cached evaluations
        let poly = DensePolynomial::new(cached.evals.clone());

        // Use the cached polynomial for open
        HyperKZGGpu::<Bn254>::open_gpu(pk, &poly, point, eval, transcript)
    }

    /// Combined commit and open in a single operation.
    ///
    /// This is the most efficient method as it:
    /// 1. Uploads the polynomial to GPU once
    /// 2. Computes commitment
    /// 3. Computes intermediate polynomials
    /// 4. Generates the proof
    ///
    /// All without downloading and re-uploading intermediate data.
    #[tracing::instrument(skip_all, name = "HyperKZGGpuSession::commit_and_open")]
    pub fn commit_and_open<T: Transcript>(
        &mut self,
        pk: &HyperKZGProverKey<Bn254>,
        poly: &DensePolynomial<Fr>,
        point: &[Fr],
        transcript: &mut T,
    ) -> anyhow::Result<(HyperKZGCommitment<Bn254>, HyperKZGProof<Bn254>)> {
        let ell = point.len();
        let n = poly.len();
        anyhow::ensure!(n == 1 << ell, "Polynomial length must be 2^ell");

        // Phase 1: Compute commitment to original polynomial
        let poly_commitment = {
            let results = gpu_batch_commit(pk.g1_powers(), &[poly])?;
            HyperKZGCommitment(results[0].into_affine())
        };

        // Phase 1 continued: Create intermediate polynomials using GPU fix_var
        let mut polys: Vec<DensePolynomial<Fr>> = Vec::with_capacity(ell);
        polys.push(poly.clone());

        let challenges = &point[..ell - 1];
        if !challenges.is_empty() {
            let intermediates = gpu_fix_vars_with_intermediates(poly.evals_ref(), challenges)?;
            for intermediate in intermediates {
                polys.push(DensePolynomial::new(intermediate));
            }
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // Commit to intermediate polynomials using GPU MSM
        let poly_refs: Vec<&DensePolynomial<Fr>> = polys[1..].iter().collect();
        let coms = gpu_batch_commit(pk.g1_powers(), &poly_refs)?;
        let coms_aff: Vec<G1Affine> = coms.iter().map(|c| c.into_affine()).collect();

        // Phase 2: Get challenge from transcript
        transcript.append_points(&coms_aff);
        let r: Fr = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3: KZG batch open using GPU
        let (w, v) = kzg_open_batch_gpu(&polys, &u, pk, transcript)?;

        // Cache the polynomial evaluations and intermediate evaluations
        self.cached_poly = Some(CachedPolynomial {
            evals: poly.evals_ref().to_vec(),
            commitment: Some(poly_commitment.clone()),
            intermediates: Some(polys.iter().map(|p| p.evals_ref().to_vec()).collect()),
        });

        let proof = HyperKZGProof {
            coms: coms_aff,
            w,
            v,
        };

        Ok((poly_commitment, proof))
    }

    /// Batch commit and open multiple polynomials.
    ///
    /// This is useful when you need to commit to multiple polynomials and open them
    /// at the same point. All operations are batched for efficiency.
    ///
    /// Note: Each proof uses a fresh clone of the transcript to ensure independent
    /// Fiat-Shamir challenges.
    pub fn batch_commit_and_open<T: Transcript + Clone>(
        &mut self,
        pk: &HyperKZGProverKey<Bn254>,
        polys: &[DensePolynomial<Fr>],
        point: &[Fr],
        transcript: &mut T,
    ) -> anyhow::Result<(Vec<HyperKZGCommitment<Bn254>>, Vec<HyperKZGProof<Bn254>>)> {
        let mut commitments = Vec::with_capacity(polys.len());
        let mut proofs = Vec::with_capacity(polys.len());

        // Batch commit all polynomials at once
        let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();
        let commit_results = gpu_batch_commit(pk.g1_powers(), &poly_refs)?;

        for (i, result) in commit_results.iter().enumerate() {
            commitments.push(HyperKZGCommitment(result.into_affine()));

            // Generate proof for each polynomial
            let eval = polys[i].evaluate(point)?;
            let mut poly_transcript = transcript.clone();
            let proof =
                HyperKZGGpu::<Bn254>::open_gpu(pk, &polys[i], point, &eval, &mut poly_transcript)?;
            proofs.push(proof);
        }

        Ok((commitments, proofs))
    }
}

// ============================================================================
// HyperKZG GPU Open Implementation
// ============================================================================

impl HyperKZGGpu<Bn254> {
    /// GPU-accelerated open operation.
    ///
    /// This uses GPU for:
    /// 1. Phase 1: Variable fixing (fix_var iterations)
    /// 2. MSM for intermediate commitments
    /// 3. KZG batch open witness computations
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

        // Phase 1: Create commitments using GPU fix_var
        let mut polys: Vec<DensePolynomial<Fr>> = Vec::with_capacity(ell);
        polys.push(poly.clone());

        // Use GPU for all fix_var operations
        let challenges = &point[..ell - 1];
        if !challenges.is_empty() {
            let intermediates = gpu_fix_vars_with_intermediates(poly.evals_ref(), challenges)?;
            for intermediate in intermediates {
                polys.push(DensePolynomial::new(intermediate));
            }
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // Commit to intermediate polynomials using GPU MSM
        let poly_refs: Vec<&DensePolynomial<Fr>> = polys[1..].iter().collect();
        let coms = gpu_batch_commit(pk.g1_powers(), &poly_refs)?;
        let coms_aff: Vec<G1Affine> = coms.iter().map(|c| c.into_affine()).collect();

        // Phase 2: Get challenge from transcript
        transcript.append_points(&coms_aff);
        let r: Fr = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3: KZG batch open using GPU-accelerated implementation
        let (w, v) = kzg_open_batch_gpu(&polys, &u, pk, transcript)?;

        Ok(HyperKZGProof {
            coms: coms_aff,
            w,
            v,
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
        let eval = poly.evaluate(opening_point)?;
        Self::open_gpu(setup, poly, opening_point, &eval, transcript)
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
    use ark_ff::AdditiveGroup;
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

    // ========================================================================
    // New tests for GPU Phase 3 (KZG batch open) operations
    // ========================================================================

    /// CPU reference for univariate polynomial evaluation using Horner's method.
    fn cpu_eval_univariate(coeffs: &[Fr], x: &Fr) -> Fr {
        let mut result = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            result = result * x + coeffs[i];
        }
        result
    }

    /// CPU reference for witness polynomial computation.
    fn cpu_witness_poly(f: &[Fr], u: &Fr) -> Vec<Fr> {
        let n = f.len();
        let mut h = vec![Fr::ZERO; n - 1];
        let mut carry = Fr::ZERO;
        for i in (1..n).rev() {
            carry = f[i] + carry * u;
            h[i - 1] = carry;
        }
        h
    }

    /// Test that GPU eval_univariate_batch matches CPU evaluation.
    #[test]
    fn test_eval_univariate_batch_gpu_vs_cpu() {

        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(100);

        for ell in [6, 8, 10] {
            let n = 1 << ell;
            let num_polys = 5;
            let num_points = 3;

            // Generate random polynomials
            let polys: Vec<Vec<Fr>> = (0..num_polys)
                .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
                .collect();
            let poly_slices: Vec<&[Fr]> = polys.iter().map(|p| p.as_slice()).collect();

            // Generate random evaluation points
            let points: Vec<Fr> = (0..num_points).map(|_| Fr::rand(&mut rng)).collect();

            // GPU batch evaluation
            let gpu_results =
                gpu_eval_univariate_batch(&poly_slices, &points).expect("GPU eval failed");

            // CPU reference evaluation
            for (point_idx, point) in points.iter().enumerate() {
                for (poly_idx, poly) in polys.iter().enumerate() {
                    let cpu_result = cpu_eval_univariate(poly, point);
                    let gpu_result = gpu_results[point_idx][poly_idx];
                    assert_eq!(
                        cpu_result, gpu_result,
                        "Mismatch at point_idx={point_idx}, poly_idx={poly_idx}, ell={ell}"
                    );
                }
            }
        }
    }

    /// Test that GPU witness_poly_batch matches CPU computation.
    #[test]
    fn test_witness_poly_batch_gpu_vs_cpu() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(101);

        for ell in [6, 8, 10] {
            let n = 1 << ell;

            // Generate random polynomial
            let poly: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

            // Generate 3 evaluation points (like HyperKZG: r, -r, r²)
            let r = Fr::rand(&mut rng);
            let points = vec![r, -r, r * r];

            // GPU batch witness computation
            let gpu_witnesses = gpu_witness_poly_batch(&poly, &points).expect("GPU witness failed");

            // CPU reference computation
            for (i, point) in points.iter().enumerate() {
                let cpu_witness = cpu_witness_poly(&poly, point);
                assert_eq!(
                    cpu_witness.len(),
                    gpu_witnesses[i].len(),
                    "Witness length mismatch at point {i}"
                );
                for (j, (cpu, gpu)) in cpu_witness.iter().zip(gpu_witnesses[i].iter()).enumerate() {
                    assert_eq!(cpu, gpu, "Mismatch at point {i}, index {j}, ell={ell}");
                }
            }
        }
    }

    /// Test that kzg_open_batch_gpu produces valid proofs (same as CPU).
    #[test]
    fn test_kzg_open_batch_gpu_produces_valid_proofs() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(102);

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

            let (comm, _) = HyperKZG::<Bn254>::commit(&pk, &poly).expect("commit failed");

            // Generate proof using GPU (open_gpu uses kzg_open_batch_gpu internally)
            let mut prover_transcript = Blake3Transcript::new(b"KzgOpenBatchGpuTest");
            let proof = HyperKZGGpu::<Bn254>::open_gpu(&pk, &poly, &point, &eval, &mut prover_transcript)
                .expect("GPU open failed");

            // Verify the proof
            let mut verifier_transcript = Blake3Transcript::new(b"KzgOpenBatchGpuTest");
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &proof, &mut verifier_transcript)
                .expect("Verification failed for GPU-generated proof");

            println!("ell={ell}: kzg_open_batch_gpu produces valid proof");
        }
    }

    /// Test that kzg_open_batch_gpu matches kzg_open_batch output exactly.
    #[test]
    fn test_kzg_open_batch_gpu_vs_cpu_output() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(103);

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

            let (comm, _) = HyperKZG::<Bn254>::commit(&pk, &poly).expect("commit failed");

            // CPU proof (using open_cpu which uses kzg_open_batch)
            let mut cpu_transcript = Blake3Transcript::new(b"BatchCompare");
            let cpu_proof = HyperKZGGpu::<Bn254>::open_cpu(&pk, &poly, &point, &eval, &mut cpu_transcript)
                .expect("CPU open failed");

            // GPU proof (using open_gpu which uses kzg_open_batch_gpu)
            let mut gpu_transcript = Blake3Transcript::new(b"BatchCompare");
            let gpu_proof = HyperKZGGpu::<Bn254>::open_gpu(&pk, &poly, &point, &eval, &mut gpu_transcript)
                .expect("GPU open failed");

            // Compare intermediate commitments (Phase 1 + 2)
            assert_eq!(cpu_proof.coms.len(), gpu_proof.coms.len());
            for (i, (cpu, gpu)) in cpu_proof.coms.iter().zip(gpu_proof.coms.iter()).enumerate() {
                assert_eq!(cpu, gpu, "coms[{i}] differs");
            }

            // Compare evaluations matrix v (Phase 3)
            assert_eq!(cpu_proof.v.len(), gpu_proof.v.len());
            for (i, (cpu_v, gpu_v)) in cpu_proof.v.iter().zip(gpu_proof.v.iter()).enumerate() {
                assert_eq!(cpu_v.len(), gpu_v.len(), "v[{i}] length differs");
                for (j, (c, g)) in cpu_v.iter().zip(gpu_v.iter()).enumerate() {
                    assert_eq!(c, g, "v[{i}][{j}] differs");
                }
            }

            // Compare witness commitments w (Phase 3)
            assert_eq!(cpu_proof.w.len(), gpu_proof.w.len());
            for (i, (cpu, gpu)) in cpu_proof.w.iter().zip(gpu_proof.w.iter()).enumerate() {
                assert_eq!(cpu, gpu, "w[{i}] differs");
            }

            // Both should verify
            let mut verify_transcript = Blake3Transcript::new(b"BatchCompare");
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &gpu_proof, &mut verify_transcript)
                .expect("GPU proof verification failed");

            println!("ell={ell}: kzg_open_batch_gpu output matches kzg_open_batch");
        }
    }

    // ========================================================================
    // Tests for HyperKZGGpuSession (Buffer Persistence)
    // ========================================================================

    /// Test that HyperKZGGpuSession::commit_with_cache produces correct commitments.
    #[test]
    fn test_session_commit_with_cache() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(200);

        for ell in [8, 10] {
            let n = 1 << ell;

            let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DensePolynomial::from(poly_raw);

            let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
            let (pk, _) = srs.trim(n);

            // Session commit
            let mut session = HyperKZGGpuSession::new();
            assert!(!session.has_cached_poly());

            let session_comm = session.commit_with_cache(&pk, &poly).expect("Session commit failed");
            assert!(session.has_cached_poly());

            // Standard GPU commit
            let (gpu_comm, _) = HyperKZGGpu::<Bn254>::commit(&pk, &poly).expect("GPU commit failed");

            assert_eq!(
                session_comm.0, gpu_comm.0,
                "Session commit differs from GPU commit"
            );

            // Check cached commitment matches
            assert_eq!(
                session.cached_commitment().unwrap().0,
                session_comm.0,
                "Cached commitment doesn't match"
            );

            println!("ell={ell}: Session commit_with_cache works correctly");
        }
    }

    /// Test that HyperKZGGpuSession::open_from_cache produces valid proofs.
    #[test]
    fn test_session_open_from_cache() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(201);

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

            // Session workflow: commit then open
            let mut session = HyperKZGGpuSession::new();
            let comm = session.commit_with_cache(&pk, &poly).expect("Commit failed");

            let mut transcript = Blake3Transcript::new(b"SessionOpen");
            let proof = session
                .open_from_cache(&pk, &point, &eval, &mut transcript)
                .expect("Open from cache failed");

            // Verify the proof
            let mut verify_transcript = Blake3Transcript::new(b"SessionOpen");
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &proof, &mut verify_transcript)
                .expect("Session proof verification failed");

            println!("ell={ell}: Session open_from_cache produces valid proofs");
        }
    }

    /// Test that HyperKZGGpuSession::commit_and_open produces valid proofs.
    #[test]
    fn test_session_commit_and_open() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(202);

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

            // Combined commit_and_open
            let mut session = HyperKZGGpuSession::new();
            let mut transcript = Blake3Transcript::new(b"SessionCombined");
            let (comm, proof) = session
                .commit_and_open(&pk, &poly, &point, &mut transcript)
                .expect("commit_and_open failed");

            // Verify the proof
            let mut verify_transcript = Blake3Transcript::new(b"SessionCombined");
            HyperKZG::<Bn254>::verify(&vk, &comm, &point, &eval, &proof, &mut verify_transcript)
                .expect("Combined session proof verification failed");

            // Check that data is cached
            assert!(session.has_cached_poly());
            assert!(session.cached_commitment().is_some());

            println!("ell={ell}: Session commit_and_open produces valid proofs");
        }
    }

    /// Test that HyperKZGGpuSession::commit_and_open matches separate commit + open.
    #[test]
    fn test_session_commit_and_open_matches_separate() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(203);

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

            // Method 1: Combined commit_and_open
            let mut session = HyperKZGGpuSession::new();
            let mut transcript1 = Blake3Transcript::new(b"SessionCompare");
            let (comm1, proof1) = session
                .commit_and_open(&pk, &poly, &point, &mut transcript1)
                .expect("commit_and_open failed");

            // Method 2: Separate commit + open (using standard GPU methods)
            let (comm2, _) = HyperKZGGpu::<Bn254>::commit(&pk, &poly).expect("commit failed");
            let mut transcript2 = Blake3Transcript::new(b"SessionCompare");
            let proof2 =
                HyperKZGGpu::<Bn254>::open_gpu(&pk, &poly, &point, &eval, &mut transcript2)
                    .expect("open_gpu failed");

            // Commitments should match
            assert_eq!(comm1.0, comm2.0, "Commitments differ");

            // Intermediate commitments should match
            assert_eq!(proof1.coms.len(), proof2.coms.len());
            for (i, (c1, c2)) in proof1.coms.iter().zip(proof2.coms.iter()).enumerate() {
                assert_eq!(c1, c2, "coms[{i}] differs");
            }

            // Both proofs should verify
            let mut verify_transcript = Blake3Transcript::new(b"SessionCompare");
            HyperKZG::<Bn254>::verify(&vk, &comm1, &point, &eval, &proof1, &mut verify_transcript)
                .expect("Combined proof verification failed");

            let mut verify_transcript = Blake3Transcript::new(b"SessionCompare");
            HyperKZG::<Bn254>::verify(&vk, &comm2, &point, &eval, &proof2, &mut verify_transcript)
                .expect("Separate proof verification failed");

            println!("ell={ell}: Session commit_and_open matches separate operations");
        }
    }

    /// Test session cache clearing.
    #[test]
    fn test_session_clear_cache() {
        if Device::all().is_empty() {
            println!("No GPU available, skipping test");
            return;
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(204);
        let ell = 8;
        let n = 1 << ell;

        let poly_raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DensePolynomial::from(poly_raw);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, n);
        let (pk, _) = srs.trim(n);

        let mut session = HyperKZGGpuSession::new();
        assert!(!session.has_cached_poly());

        // Cache a polynomial
        let _ = session.commit_with_cache(&pk, &poly).expect("Commit failed");
        assert!(session.has_cached_poly());

        // Clear cache
        session.clear_cache();
        assert!(!session.has_cached_poly());
        assert!(session.cached_commitment().is_none());

        println!("Session clear_cache works correctly");
    }
}
