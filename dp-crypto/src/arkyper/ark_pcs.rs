use std::marker::PhantomData;

use anyhow::ensure;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_poly_commit::multilinear_pc::{MultilinearPC, data_structures::CommitterKey};
use ark_std::cfg_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::dense::DensePolynomial,
};

#[derive(Default, Clone)]
pub struct ArkPcs<P: Pairing>(PhantomData<MultilinearPC<P>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverKey<P: Pairing>(#[serde(with = "crate::serialization")] CommitterKey<P>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierKey<P: Pairing>(
    #[serde(with = "crate::serialization")]
    ark_poly_commit::multilinear_pc::data_structures::VerifierKey<P>,
);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment<P: Pairing>(
    #[serde(with = "crate::serialization")]
    ark_poly_commit::multilinear_pc::data_structures::Commitment<P>,
);

impl<P: Pairing> PartialEq for Commitment<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0.nv == other.0.nv && self.0.g_product == other.0.g_product
    }
}

impl<P: Pairing> Default for Commitment<P> {
    fn default() -> Self {
        Commitment(
            ark_poly_commit::multilinear_pc::data_structures::Commitment {
                nv: 0,
                g_product: P::G1Affine::zero(),
            },
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof<P: Pairing>(
    #[serde(with = "crate::serialization")]
    ark_poly_commit::multilinear_pc::data_structures::Proof<P>,
);

impl<P: Pairing> AppendToTranscript for Commitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_bytes(self.0.nv.to_le_bytes().as_slice());
        transcript.append_points(&[self.0.g_product]);
    }
}

impl<P: Pairing> CommitmentScheme for ArkPcs<P> {
    type Field = P::ScalarField;

    type ProverSetup = ProverKey<P>;

    type VerifierSetup = VerifierKey<P>;

    type Commitment = Commitment<P>;

    type Proof = Proof<P>;

    type BatchedProof = Proof<P>;

    type OpeningProofHint = ();

    fn test_setup<R: ark_std::rand::Rng + ark_std::rand::RngCore>(
        rng: &mut R,
        max_num_vars: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        let params = MultilinearPC::<P>::setup(max_num_vars, rng);
        let (ck, vk) = MultilinearPC::trim(&params, max_num_vars);
        (ProverKey(ck), VerifierKey(vk))
    }

    fn commit(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)> {
        Ok((Commitment(MultilinearPC::commit(&setup.0, poly)), ()))
    }

    fn batch_commit<'a, U>(
        gens: &Self::ProverSetup,
        polys: &[U],
    ) -> anyhow::Result<Vec<(Self::Commitment, Self::OpeningProofHint)>>
    where
        U: std::borrow::Borrow<DensePolynomial<'a, Self::Field>> + Sync,
    {
        cfg_iter!(polys)
            .map(|poly| Self::commit(gens, poly.borrow()))
            .collect::<anyhow::Result<_>>()
    }

    fn combine_commitments<C: std::borrow::Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> anyhow::Result<Self::Commitment> {
        let combined_commitment: P::G1 = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.borrow().0.g_product * coeff)
            .sum();
        let nv = commitments
            .iter()
            .map(|comm| comm.borrow().0.nv)
            .max()
            .unwrap_or_default();
        Ok(Commitment(
            ark_poly_commit::multilinear_pc::data_structures::Commitment {
                nv,
                g_product: combined_commitment.into_affine(),
            },
        ))
    }

    fn prove<ProofTranscript: super::transcript::Transcript>(
        setup: &Self::ProverSetup,
        poly: &crate::poly::dense::DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> anyhow::Result<Self::Proof> {
        Ok(Proof(MultilinearPC::open(&setup.0, poly, opening_point)))
    }

    fn verify<ProofTranscript: super::transcript::Transcript>(
        setup: &Self::VerifierSetup,
        proof: &Self::Proof,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> anyhow::Result<()> {
        ensure!(
            MultilinearPC::check(&setup.0, &commitment.0, opening_point, *opening, &proof.0,),
            "MultilinearPC proof verification failed"
        );
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"ark-multilinear-pcs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arkyper::transcript::blake3::Blake3Transcript, poly::challenge};
    use ark_bn254::{Bn254, Fr};
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::{
        UniformRand,
        rand::{SeedableRng, thread_rng},
    };
    use itertools::Itertools;

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

            let (pk, vk) = ArkPcs::<Bn254>::test_setup(&mut rng, ell);

            // make a commitment
            let (comm, _) = ArkPcs::commit(&pk, &poly).unwrap();

            // prove an evaluation
            let mut prover_transcript = Blake3Transcript::new(b"TestEval");
            let proof = ArkPcs::prove(&pk, &poly, &point, None, &mut prover_transcript).unwrap();

            // verify the evaluation
            let mut verifier_tr = Blake3Transcript::new(b"TestEval");
            ArkPcs::verify(&vk, &proof, &mut verifier_tr, &point, &eval, &comm).unwrap();
            // Change the proof and expect verification to fail
            let mut bad_proof = proof.clone();
            let v1 = bad_proof.0.proofs[1];
            bad_proof.0.proofs[0].clone_from(&v1);
            let mut verifier_tr2 = Blake3Transcript::new(b"TestEval");
            assert!(
                ArkPcs::verify(&vk, &bad_proof, &mut verifier_tr2, &point, &eval, &comm).is_err()
            );
        }
    }

    #[test]
    fn test_batch_open() -> anyhow::Result<()> {
        const NUM_VARS: &[usize] = &[12, 14, 16];

        let num_polys = NUM_VARS.len();
        let max_num_vars = *NUM_VARS.iter().max().unwrap();

        let rng = &mut thread_rng();

        // generate polynomials
        let polys = NUM_VARS
            .iter()
            .map(|num_vars| DensePolynomial::random(*num_vars, rng))
            .collect_vec();

        let (pp, vp) = ArkPcs::<Bn254>::test_setup(rng, max_num_vars);

        let commitments = ArkPcs::batch_commit(&pp, &polys)?
            .into_iter()
            .map(|(commitment, _)| commitment)
            .collect_vec();

        let opening_point = (0..max_num_vars)
            .map(|_| challenge::random_challenge::<Fr, _>(rng))
            .collect_vec();

        let evals = polys
            .iter()
            .map(|poly| poly.evaluate(&opening_point[..poly.num_vars()]))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let coefficients = (0..num_polys)
            .map(|_| challenge::random_challenge::<Fr, _>(rng))
            .collect_vec();

        let rlc_poly = DensePolynomial::linear_combination(
            polys.iter().collect_vec().as_slice(),
            &coefficients,
        );
        let mut transcript = Blake3Transcript::new(b"batch_open");

        let proof = ArkPcs::prove(&pp, &rlc_poly, &opening_point, None, &mut transcript)?;

        let comm = ArkPcs::combine_commitments(&commitments, &coefficients)?;

        let eval = evals.into_iter().zip(coefficients).enumerate().fold(
            Fr::ZERO,
            |eval, (i, (ev, coeff))| {
                let factor = (NUM_VARS[i]..max_num_vars).fold(Fr::ONE, |factor, var_index| {
                    factor * (Fr::ONE - opening_point[var_index])
                });
                eval + ev * coeff * factor
            },
        );

        assert_eq!(eval, rlc_poly.evaluate(&opening_point)?);

        let mut transcript = Blake3Transcript::new(b"batch_open");

        ArkPcs::verify(&vp, &proof, &mut transcript, &opening_point, &eval, &comm)
    }
}
