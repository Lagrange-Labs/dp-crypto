//! This is a port of https://github.com/microsoft/Nova/blob/main/src/provider/hyperkzg.rs
//! and such code is Copyright (c) Microsoft Corporation.
//!
//! This module implements `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
use crate::poly::dense::DensePolynomial;
use anyhow::{bail, ensure};
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_poly_commit::kzg10::{KZG10, Powers, UniversalParams, VerifierKey};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use ark_std::{One, Zero, cfg_iter, cfg_iter_mut};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::borrow::Borrow;
use std::marker::PhantomData;
pub mod interface;
pub mod msm;
pub mod transcript;
pub use interface::*;
use transcript::Transcript;

// just a type needed to create the SRS
#[allow(dead_code)]
type UniPoly<P> = ark_poly::polynomial::univariate::DensePolynomial<<P as Pairing>::ScalarField>;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGSRS<P: Pairing>(UniversalParams<P>);

impl<P: Pairing> HyperKZGSRS<P> {
    pub fn setup<R: Rng + RngCore>(rng: &mut R, max_degree: usize) -> Self {
        let params = KZG10::<P, UniPoly<P>>::setup(max_degree, true, rng).unwrap();
        Self(params)
    }

    pub fn trim(mut self, mut max_degree: usize) -> (HyperKZGProverKey<P>, HyperKZGVerifierKey<P>) {
        let (kzg_pk, kzg_vk) = {
            // currentl logic only lives in test, extracted from
            // https://github.com/arkworks-rs/poly-commit/blob/a05ec99d0d3e46c8ba0d6d3592980777bc2847e8/poly-commit/src/kzg10/mod.rs#L491
            if max_degree == 1 {
                max_degree += 1;
            }
            self.0
                .powers_of_g
                .resize(max_degree + 1, P::G1Affine::zero());
            let powers_of_gamma_g = (0..=max_degree)
                .map(|i| self.0.powers_of_gamma_g[&i])
                .collect();
            let vk = ark_poly_commit::kzg10::VerifierKey {
                g: self.0.powers_of_g[0],
                gamma_g: self.0.powers_of_gamma_g[&0],
                h: self.0.h,
                beta_h: self.0.beta_h,
                prepared_h: self.0.prepared_h.clone(),
                prepared_beta_h: self.0.prepared_beta_h.clone(),
            };
            let powers = ark_poly_commit::kzg10::Powers {
                powers_of_g: ark_std::borrow::Cow::Owned(self.0.powers_of_g),
                powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
            };

            (powers, vk)
        };
        (HyperKZGProverKey { kzg_pk }, HyperKZGVerifierKey { kzg_vk })
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGProverKey<P: Pairing> {
    pub kzg_pk: Powers<'static, P>,
}

impl<P: Pairing> HyperKZGProverKey<P> {
    pub fn g1_powers(&self) -> &[P::G1Affine] {
        &self.kzg_pk.powers_of_g
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGVerifierKey<P: Pairing> {
    pub kzg_vk: VerifierKey<P>,
}

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGCommitment<P: Pairing>(pub P::G1Affine);

impl<P: Pairing> Default for HyperKZGCommitment<P> {
    fn default() -> Self {
        Self(P::G1Affine::zero())
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperKZGProof<P: Pairing> {
    pub coms: Vec<P::G1Affine>,
    pub w: Vec<P::G1Affine>,
    pub v: Vec<Vec<P::ScalarField>>,
}

// On input f(x) and u compute the witness polynomial used to prove
// that f(u) = v. The main part of this is to compute the
// division (f(x) - f(u)) / (x - u), but we don't use a general
// division algorithm, we make use of the fact that the division
// never has a remainder, and that the denominator is always a linear
// polynomial. The cost is (d-1) mults + (d-1) adds in P::ScalarField, where
// d is the degree of f.
//
// We use the fact that if we compute the quotient of f(x)/(x-u),
// there will be a remainder, but it'll be v = f(u).  Put another way
// the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
// same.  One advantage is that computing f(u) could be decoupled
// from kzg_open, it could be done later or separate from computing W.
fn kzg_batch_open_no_rem<P: Pairing>(
    f: &DensePolynomial<P::ScalarField>,
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
) -> anyhow::Result<Vec<P::G1>> {
    let polys = cfg_iter!(u)
        .map(|ui| {
            let h = compute_witness_polynomial::<P>(&f.evals(), *ui);
            DensePolynomial::new(h)
        })
        .collect::<Vec<_>>();

    let g1_powers = &pk.kzg_pk.powers_of_g;

    // batch commit requires all batches to have the same length
    ensure!(polys.iter().all(|s| s.len() == polys[0].len()));

    if let Some((i, invalid)) = polys
        .iter()
        .enumerate()
        .find(|(_, coeffs)| (*coeffs).len() > g1_powers.len())
    {
        bail!(
            "Invalid {}-th polynomial length -> len {} vs pp powers {}",
            i,
            invalid.len(),
            g1_powers.len()
        );
    }

    let msm_size = polys[0].borrow().len();
    msm::batch_poly_msm(&g1_powers[..msm_size], &polys)
}

fn compute_witness_polynomial<P: Pairing>(
    f: &[P::ScalarField],
    u: P::ScalarField,
) -> Vec<P::ScalarField> {
    let d = f.len();

    // Compute h(x) = f(x)/(x - u)
    let mut h = vec![P::ScalarField::zero(); d];
    for i in (1..d).rev() {
        h[i - 1] = f[i] + h[i] * u;
    }

    h
}

#[allow(clippy::type_complexity)]
fn kzg_open_batch<P: Pairing, T: Transcript>(
    f: &[DensePolynomial<P::ScalarField>],
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
    transcript: &mut T,
) -> anyhow::Result<(Vec<P::G1Affine>, Vec<Vec<P::ScalarField>>)> {
    let k = f.len();
    let t = u.len();

    // The verifier needs f_i(u_j), so we compute them here
    // (V will compute B(u_j) itself)
    let mut v = vec![vec!(P::ScalarField::zero(); k); t];
    cfg_iter_mut!(v).enumerate().for_each(|(i, v_i)| {
        // for each point u
        #[cfg(feature = "parallel")]
        let it = v_i.par_iter_mut().zip_eq(f);
        #[cfg(not(feature = "parallel"))]
        let it = v_i.iter_mut().zip(f);

        it.for_each(|(v_ij, f)| {
            // for each poly f
            *v_ij = f.eval_as_univariate(&u[i]);
        });
    });

    // TODO(moodlezoup): Avoid cloned()
    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(&scalars);
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(f.len());

    let b_poly = {
        let poly_refs: Vec<&DensePolynomial<P::ScalarField>> = f.iter().collect();
        DensePolynomial::linear_combination(&poly_refs, &q_powers)
    };

    // Now open B at u0, ..., u_{t-1}
    let w = kzg_batch_open_no_rem(&b_poly, u, pk)?;
    let w_aff = w
        .iter()
        .map(|g| g.into_affine())
        .collect::<Vec<P::G1Affine>>();

    // The prover computes the challenge to keep the transcript in the same
    // state as that of the verifier
    transcript.append_points(&w_aff);
    let _d_0: P::ScalarField = transcript.challenge_scalar();

    Ok((w_aff, v))
}

// vk is hashed in transcript already, so we do not add it here
fn kzg_verify_batch<P: Pairing, ProofTranscript: Transcript>(
    vk: &HyperKZGVerifierKey<P>,
    c_points: &[P::G1Affine],
    w_points: &[P::G1Affine],
    u: &[P::ScalarField],
    v: &[Vec<P::ScalarField>],
    transcript: &mut ProofTranscript,
) -> bool {
    let k = c_points.len();
    let t = u.len();

    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(scalars.as_slice());
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(k);

    transcript.append_points(w_points);
    let d_0: P::ScalarField = transcript.challenge_scalar();
    let d_1 = d_0 * d_0;

    assert_eq!(t, 3);
    assert_eq!(w_points.len(), 3);
    // We write a special case for t=3, since this what is required for
    // hyperkzg. Following the paper directly, we must compute:
    // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
    // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
    // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
    // let R0 = -W[0];
    // let R1 = -W[1];
    // let R2 = -W[2];
    // let L = L0 + L1*d_0 + L2*d_1;
    // let R = R0 + R1*d_0 + R2*d_1;
    //
    // We group terms to reduce the number of scalar mults (to seven):
    // In Rust, we could use MSMs for these, and speed up verification.
    //
    // Note, that while computing L, the intermediate computation of C_B together with computing
    // L0, L1, L2 can be replaced by single MSM of C with the powers of q multiplied by (1 + d_0 + d_1)
    // with additionally concatenated inputs for scalars/bases.

    let q_power_multiplier: P::ScalarField = P::ScalarField::one() + d_0 + d_1;

    let q_powers_multiplied: Vec<P::ScalarField> = q_powers
        .par_iter()
        .with_min_len(4096)
        .map(|q_power| q_power_multiplier * q_power)
        .collect();

    // Compute the batched openings
    // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
    let b_u = v
        .into_par_iter()
        .map(|v_i| {
            v_i.into_par_iter()
                .zip(q_powers.par_iter())
                .with_min_len(4096)
                .map(|(a, b)| *a * *b)
                .sum()
        })
        .collect::<Vec<P::ScalarField>>();

    let l = msm::msm(
        &[
            &c_points[..k],
            &[w_points[0], w_points[1], w_points[2], vk.kzg_vk.g],
        ]
        .concat(),
        &[
            &q_powers_multiplied[..k],
            &[
                u[0],
                (u[1] * d_0),
                (u[2] * d_1),
                -(b_u[0] + d_0 * b_u[1] + d_1 * b_u[2]),
            ],
        ]
        .concat(),
    )
    .unwrap();

    let r = w_points[0] + w_points[1] * d_0 + w_points[2] * d_1;

    // Check that e(L, vk.H) == e(R, vk.tau_H)
    P::multi_pairing([l, -r], [vk.kzg_vk.h, vk.kzg_vk.beta_h]).is_zero()
}

#[derive(Clone)]
pub struct HyperKZG<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> HyperKZG<P> {
    #[tracing::instrument(skip_all, name = "HyperKZG::open")]
    pub fn open<ProofTranscript: Transcript>(
        pk: &HyperKZGProverKey<P>,
        poly: &DensePolynomial<P::ScalarField>,
        point: &[P::ScalarField],
        _eval: &P::ScalarField,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<HyperKZGProof<P>> {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

        // Phase 1  -- create commitments com_1, ..., com_\ell
        // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
        // also known to verifier, so can be derived on its side as well
        let mut polys: Vec<DensePolynomial<P::ScalarField>> = Vec::new();
        polys.push(poly.clone());
        for i in 0..ell - 1 {
            let previous_poly: &DensePolynomial<P::ScalarField> = &polys[i];
            let pi_len = previous_poly.len() / 2;
            let indices = (0..pi_len).collect::<Vec<usize>>();
            let coeffs = cfg_iter!(indices)
                .map(|j| {
                    point[ell - i - 1] * (previous_poly[2 * j + 1] - previous_poly[2 * j])
                        + previous_poly[2 * j]
                })
                .collect();
            polys.push(DensePolynomial::new(coeffs));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        println!(
            "polys.all().len(): {:?}",
            polys.iter().map(|p| p.len()).collect::<Vec<usize>>()
        );
        println!("powers.len(): {:?}", pk.g1_powers().len());
        // We do not need to commit to the first polynomial as it is already committed.
        let coms = msm::batch_poly_msm(pk.g1_powers(), &polys[1..])?;
        let coms_aff = coms
            .iter()
            .map(|c| c.into_affine())
            .collect::<Vec<P::G1Affine>>();

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        transcript.append_points(&coms_aff);
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3 -- create response
        let (w, v) = kzg_open_batch(&polys, &u, pk, transcript)?;

        Ok(HyperKZGProof {
            coms: coms_aff,
            w,
            v,
        })
    }

    /// A method to verify purported evaluations of a batch of polynomials
    pub fn verify<ProofTranscript: Transcript>(
        vk: &HyperKZGVerifierKey<P>,
        comm: &HyperKZGCommitment<P>,
        point: &[P::ScalarField],
        px: &P::ScalarField,
        pi: &HyperKZGProof<P>,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<()> {
        let y = px;

        let ell = point.len();

        let mut coms = pi.coms.clone();

        // we do not need to add x to the transcript, because in our context x was
        // obtained from the transcript
        transcript.append_points(&coms);
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();

        ensure!(!(r == P::ScalarField::zero() || comm.0 == P::G1Affine::zero()));
        coms.insert(0, comm.0); // set com_0 = C, shifts other commitments to the right

        let u = vec![r, -r, r * r];

        // Setup vectors (Y, ypos, yneg) from pi.v
        let v = &pi.v;
        ensure!(v.len() == 3);
        ensure!(v[0].len() == ell && v[1].len() == ell && v[2].len() == ell);
        let ypos = &v[0];
        let yneg = &v[1];
        let mut y_list = v[2].to_vec();
        y_list.push(*y);

        // Check consistency of (Y, ypos, yneg)
        let two = P::ScalarField::from(2u64);
        for i in 0..ell {
            let left = two * r * y_list[i + 1];
            let right = r * (P::ScalarField::one() - point[ell - i - 1]) * (ypos[i] + yneg[i])
                + point[ell - i - 1] * (ypos[i] - yneg[i]);
            ensure!(left == right);
            // Note that we don't make any checks about Y[0] here, but our batching
            // check below requires it
        }

        // Check commitments to (Y, ypos, yneg) are valid
        ensure!(
            kzg_verify_batch(vk, &coms, &pi.w, &u, &pi.v, transcript),
            "verification failed"
        );
        Ok(())
    }
}

impl<P: Pairing> CommitmentScheme for HyperKZG<P> {
    type Field = P::ScalarField;
    type ProverSetup = HyperKZGProverKey<P>;
    type VerifierSetup = HyperKZGVerifierKey<P>;

    type Commitment = HyperKZGCommitment<P>;
    type Proof = HyperKZGProof<P>;
    type BatchedProof = HyperKZGProof<P>;
    type OpeningProofHint = ();

    fn test_setup<R: Rng + RngCore>(
        rng: &mut R,
        max_num_vars: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        HyperKZGSRS::setup(rng, 1 << max_num_vars).trim(1 << max_num_vars)
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::commit")]
    fn commit(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)> {
        assert!(
            setup.g1_powers().len() >= poly.len(),
            "COMMIT KEY LENGTH ERROR {}, {}",
            setup.g1_powers().len(),
            poly.len()
        );
        let comm = HyperKZGCommitment(msm::poly_msm(setup.g1_powers(), poly)?.into_affine());
        Ok((comm, ()))
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::batch_commit")]
    fn batch_commit<U>(
        gens: &Self::ProverSetup,
        polys: &[U],
    ) -> anyhow::Result<Vec<(Self::Commitment, Self::OpeningProofHint)>>
    where
        U: Borrow<DensePolynomial<Self::Field>> + Sync,
    {
        Ok(msm::batch_poly_msm(gens.g1_powers(), polys)?
            .into_iter()
            .map(|c| (HyperKZGCommitment(c.into_affine()), ()))
            .collect())
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> anyhow::Result<Self::Commitment> {
        let combined_commitment: P::G1 = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.borrow().0 * coeff)
            .sum();
        Ok(HyperKZGCommitment(combined_commitment.into_affine()))
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<Self::Proof> {
        let eval = poly.evaluate(opening_point)?;
        HyperKZG::<P>::open(setup, poly, opening_point, &eval, transcript)
    }

    fn verify<ProofTranscript: Transcript>(
        setup: &Self::VerifierSetup,
        proof: &Self::Proof,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> anyhow::Result<()> {
        HyperKZG::<P>::verify(setup, commitment, opening_point, opening, proof, transcript)
    }

    fn protocol_name() -> &'static [u8] {
        b"arkyper-kzg"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arkyper::transcript::blake3::Blake3Transcript, poly::challenge};
    use ark_bn254::{Bn254, Fr};
    use ark_std::{UniformRand, rand::SeedableRng};

    #[test]
    fn test_hyperkzg_large() {
        // test the hyperkzg prover and verifier with random instances (derived from a seed)
        for ell in [8, 9, 10] {
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(ell as u64);

            let n = 1 << ell; // n = 2^ell

            let poly_raw = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let poly = DensePolynomial::from(poly_raw.clone());
            let point = (0..ell)
                .map(|_| challenge::random_challenge::<Fr, _>(&mut rng))
                .collect::<Vec<_>>();
            let eval = poly.evaluate(&point).unwrap();

            let srs = HyperKZGSRS::setup(&mut rng, n);
            let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(n);

            // make a commitment
            let (comm, _) = HyperKZG::commit(&pk, &poly).unwrap();

            // prove an evaluation
            let mut prover_transcript = Blake3Transcript::new(b"TestEval");
            let proof: HyperKZGProof<Bn254> =
                HyperKZG::open(&pk, &poly, &point, &eval, &mut prover_transcript).unwrap();

            // verify the evaluation
            let mut verifier_tr = Blake3Transcript::new(b"TestEval");
            assert!(HyperKZG::verify(&vk, &comm, &point, &eval, &proof, &mut verifier_tr,).is_ok());

            // Change the proof and expect verification to fail
            let mut bad_proof = proof.clone();
            let v1 = bad_proof.v[1].clone();
            bad_proof.v[0].clone_from(&v1);
            let mut verifier_tr2 = Blake3Transcript::new(b"TestEval");
            assert!(
                HyperKZG::verify(&vk, &comm, &point, &eval, &bad_proof, &mut verifier_tr2,)
                    .is_err()
            );
        }
    }
}
