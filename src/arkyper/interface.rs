use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{Rng, RngCore};
use std::{borrow::Borrow, fmt::Debug};

use crate::{arkyper::Transcript, poly::dense::DensePolynomial};

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: ark_ff::Field + Sized;
    type ProverSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type VerifierSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type Commitment: Default
        + Debug
        + Sync
        + Send
        + PartialEq
        + CanonicalSerialize
        + CanonicalDeserialize
        + Clone;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize + Clone + Debug;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    /// A hint that helps the prover compute an opening proof. Typically some byproduct of
    /// the commitment computation, e.g. for Dory the Pedersen commitments to the rows can be
    /// used as a hint for the opening proof.
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;

    /// Generates the prover setup for this PCS. `max_num_vars` is the maximum number of
    /// variables of any polynomial that will be committed using this setup.
    fn test_setup<R: Rng + RngCore>(
        rng: &mut R,
        max_num_vars: usize,
    ) -> (Self::ProverSetup, Self::VerifierSetup);

    /// Commits to a multilinear polynomial using the provided setup.
    ///
    /// # Arguments
    /// * `poly` - The multilinear polynomial to commit to
    /// * `setup` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A tuple containing the commitment to the polynomial and a hint that can be used
    /// to optimize opening proof generation
    fn commit(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
    ) -> anyhow::Result<(Self::Commitment, Self::OpeningProofHint)>;

    /// Commits to multiple multilinear polynomials in batch.
    ///
    /// # Arguments
    /// * `polys` - A slice of multilinear polynomials to commit to
    /// * `gens` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A vector of commitments, one for each input polynomial
    fn batch_commit<'a, U>(
        gens: &Self::ProverSetup,
        polys: &[U],
    ) -> anyhow::Result<Vec<(Self::Commitment, Self::OpeningProofHint)>>
    where
        U: Borrow<DensePolynomial<'a, Self::Field>> + Sync;

    /// Homomorphically combines multiple commitments into a single commitment, computed as a
    /// linear combination with the given coefficients.
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> anyhow::Result<Self::Commitment>;

    /// Homomorphically combines multiple opening proof hints into a single hint, computed as a
    /// linear combination with the given coefficients.
    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        unimplemented!()
    }

    /// Generates a proof of evaluation for a polynomial at a specific point.
    ///
    /// # Arguments
    /// * `setup` - The prover setup for the commitment scheme
    /// * `poly` - The multilinear polynomial being proved
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `hint` - An optional hint that helps optimize the proof generation.
    ///   When `None`, implementations should compute the hint internally if needed.
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    ///
    /// # Returns
    /// A proof of the polynomial evaluation at the specified point
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> anyhow::Result<Self::Proof>;

    /// Verifies a proof of polynomial evaluation at a specific point.
    ///
    /// # Arguments
    /// * `proof` - The proof to be verified
    /// * `setup` - The verifier setup for the commitment scheme
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `opening` - The claimed evaluation value of the polynomial at the opening point
    /// * `commitment` - The commitment to the polynomial
    ///
    /// # Returns
    /// Ok(()) if the proof is valid, otherwise a ProofVerifyError
    fn verify<ProofTranscript: Transcript>(
        setup: &Self::VerifierSetup,
        proof: &Self::Proof,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> anyhow::Result<()>;

    fn protocol_name() -> &'static [u8];
}
