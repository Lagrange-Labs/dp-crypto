use ark_ff::PrimeField;

use crate::{
    arkyper::transcript::Transcript,
    extrapolate::extrapolate_uni_poly,
    structs::{IOPProof, IOPProverMessage, IOPVerifierState, SumCheckSubClaim},
    virtual_poly::VPAuxInfo,
};

impl<F: PrimeField> IOPVerifierState<F> {
    pub fn verify(
        claimed_sum: F,
        proof: &IOPProof<F>,
        aux_info: &VPAuxInfo<F>,
        transcript: &mut impl Transcript,
    ) -> SumCheckSubClaim<F> {
        if aux_info.max_num_variables == 0 {
            return SumCheckSubClaim {
                point: vec![],
                expected_evaluation: claimed_sum,
            };
        }
        let start = entered_span!("sum check verify");

        transcript.append_bytes(&aux_info.max_num_variables.to_le_bytes());
        transcript.append_bytes(&aux_info.max_degree.to_le_bytes());

        let mut verifier_state = IOPVerifierState::verifier_init(aux_info);
        for i in 0..aux_info.max_num_variables {
            let prover_msg = proof.proofs.get(i).expect("proof is incomplete");
            prover_msg
                .evaluations
                .iter()
                .for_each(|e| transcript.append_scalars::<F>(&[e]));
            Self::verify_round_and_update_state(&mut verifier_state, prover_msg, transcript);
        }

        let res = Self::check_and_generate_subclaim(&verifier_state, &claimed_sum);

        exit_span!(start);
        res
    }

    /// Initialize the verifier's state.
    pub fn verifier_init(index_info: &VPAuxInfo<F>) -> Self {
        let start = entered_span!("sum check verifier init");
        let verifier_state = Self {
            round: 1,
            num_vars: index_info.max_num_variables,
            max_degree: index_info.max_degree,
            finished: false,
            polynomials_received: Vec::with_capacity(index_info.max_num_variables),
            challenges: Vec::with_capacity(index_info.max_num_variables),
        };
        exit_span!(start);
        verifier_state
    }

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    pub(crate) fn verify_round_and_update_state(
        &mut self,
        prover_msg: &IOPProverMessage<F>,
        transcript: &mut impl Transcript,
    ) -> F {
        let start = entered_span!("sum check verify {}-th round and update state", self.round);

        assert!(
            !self.finished,
            "Incorrect verifier state: Verifier is already finished."
        );

        // In an interactive protocol, the verifier should
        //
        // 1. check if the received 'P(0) + P(1) = expected`.
        // 2. set `expected` to P(r)`
        //
        // When we turn the protocol to a non-interactive one, it is sufficient to defer
        // such checks to `check_and_generate_subclaim` after the last round.

        let challenge = transcript.append_and_sample(b"Internal round");
        self.challenges.push(challenge);
        self.polynomials_received
            .push(prover_msg.evaluations.to_vec());

        if self.round == self.num_vars {
            // accept and close
            self.finished = true;
        } else {
            // proceed to the next round
            self.round += 1;
        }

        exit_span!(start);
        challenge
    }

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    pub(crate) fn check_and_generate_subclaim(&self, asserted_sum: &F) -> SumCheckSubClaim<F> {
        let start = entered_span!("sum check check and generate subclaim");
        if !self.finished {
            panic!("Incorrect verifier state: Verifier has not finished.",);
        }

        if self.polynomials_received.len() != self.num_vars {
            panic!("insufficient rounds",);
        }

        // the deferred check during the interactive phase:
        // 2. set `expected` to P(r)`
        let mut expected_vec = self
            .polynomials_received
            .iter()
            .zip(self.challenges.iter())
            .map(|(evaluations, challenge)| {
                if evaluations.len() != self.max_degree + 1 {
                    panic!(
                        "incorrect number of evaluations: {} vs {}",
                        evaluations.len(),
                        self.max_degree + 1
                    );
                }
                extrapolate_uni_poly::<F>(evaluations, *challenge)
            })
            .collect::<Vec<_>>();

        // l-append asserted_sum to the first position of the expected vector
        expected_vec.insert(0, *asserted_sum);

        for (i, (evaluations, &expected)) in self
            .polynomials_received
            .iter()
            .zip(expected_vec.iter())
            .enumerate()
            .take(self.num_vars)
        {
            // the deferred check during the interactive phase:
            // 1. check if the received 'P(0) + P(1) = expected`.
            if evaluations[0] + evaluations[1] != expected {
                panic!(
                    "{}th round's prover message is not consistent with the claim. {:?} {:?}",
                    i,
                    evaluations[0] + evaluations[1],
                    expected
                );
            }
        }
        exit_span!(start);
        SumCheckSubClaim {
            point: self.challenges.clone(),
            // the last expected value (not checked within this function) will be included in the
            // subclaim
            expected_evaluation: expected_vec[self.num_vars],
        }
    }
}
