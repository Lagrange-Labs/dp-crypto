use ark_std::rand::{Rng, RngCore};
use ark_std::vec;
use std::{borrow::Borrow, fmt::Debug};

use crate::{
    arkyper::{CommitmentScheme, Transcript, transcript::AppendToTranscript},
    poly::dense::DensePolynomial,
};

#[derive(Clone, Debug, Default)]
pub struct MockCommitmentScheme<F> {
    _marker: std::marker::PhantomData<F>,
}

impl AppendToTranscript for () {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
    }
}

impl<F: ark_ff::Field + Sized> CommitmentScheme for MockCommitmentScheme<F> {
    type Field = F;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = ();
    type Proof = ();
    type BatchedProof = ();
    type OpeningProofHint = ();

    fn test_setup<R: Rng + RngCore>(
        _: &mut R,
        _: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        ((), ())
    }

    fn commit(
        _: &Self::ProverSetup,
        _: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)> {
        Ok(((), ()))
    }

    fn batch_commit<'a, U>(
        _: &Self::ProverSetup,
        _: &[U],
    ) -> anyhow::Result<Vec<(Self::Commitment, Self::OpeningProofHint)>>
    where
        U: Borrow<DensePolynomial<'a, Self::Field>> + Sync,
    {
        Ok(vec![((), ()); 0])
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _: &[C],
        _: &[Self::Field],
    ) -> anyhow::Result<Self::Commitment> {
        Ok(())
    }

    fn combine_hints(_: Vec<Self::OpeningProofHint>, _: &[Self::Field]) -> Self::OpeningProofHint {}

    fn prove<ProofTranscript: Transcript>(
        _: &Self::ProverSetup,
        _: &DensePolynomial<Self::Field>,
        _: &[Self::Field],
        _: Option<Self::OpeningProofHint>,
        _: &mut ProofTranscript,
    ) -> anyhow::Result<Self::Proof> {
        Ok(())
    }

    fn verify<ProofTranscript: Transcript>(
        _: &Self::VerifierSetup,
        _: &Self::Proof,
        _: &mut ProofTranscript,
        _: &[Self::Field],
        _: &Self::Field,
        _: &Self::Commitment,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"MockCommitmentScheme"
    }
}
#[cfg(test)]
mod tests {
    use crate::arkyper::transcript::blake3;

    use super::*;
    use ark_bn254::Fr as F;
    use ark_ff::Field;
    use ark_std::rand::thread_rng;

    #[test]
    fn test_mock_commitment_scheme() {
        let mut rng = thread_rng();
        let (prover_setup, verifier_setup) = MockCommitmentScheme::<F>::test_setup(&mut rng, 0);
        let polynomial = DensePolynomial::new(
            vec![1, 2, 3, 4]
                .into_iter()
                .map(F::from)
                .collect::<Vec<_>>(),
        );

        // Test commit
        let (commitment, hint) =
            MockCommitmentScheme::<F>::commit(&prover_setup, &polynomial).unwrap();

        // Test batch_commit
        let _batch_commitments =
            MockCommitmentScheme::<F>::batch_commit(&prover_setup, &[&polynomial]).unwrap();

        // Test combine_commitments
        let _combined_commitment =
            MockCommitmentScheme::<F>::combine_commitments(&[&commitment], &[F::ONE]).unwrap();

        // Test combine_hints
        let combined_hint = MockCommitmentScheme::<F>::combine_hints(vec![hint], &[F::ONE]);

        let mut transcript = blake3::Blake3Transcript::new(b"test");
        // Test prove
        let proof = MockCommitmentScheme::<F>::prove(
            &prover_setup,
            &polynomial,
            &[F::ONE],
            Some(combined_hint),
            &mut transcript,
        )
        .unwrap();

        // Test verify
        let verify_result = MockCommitmentScheme::<F>::verify(
            &verifier_setup,
            &proof,
            &mut transcript,
            &[F::ONE],
            &F::ONE,
            &commitment,
        );
        assert!(verify_result.is_ok());

        // Test protocol_name
        let name = MockCommitmentScheme::<F>::protocol_name();
        assert_eq!(name, b"MockCommitmentScheme");
    }
}
