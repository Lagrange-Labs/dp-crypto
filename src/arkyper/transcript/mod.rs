use std::borrow::Borrow;

use ark_ec::{AffineRepr};
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;
pub mod blake3;

/// Constant serialization of an infinity point. 
const ZERO_POINT_SERIALIZED: [u8; 16] = [0u8; 16];

// TODO: add the string DST on each append()
pub trait Transcript {
    fn append_bytes(&mut self, bytes: &[u8]);
    fn challenge_bytes(&mut self, out: &mut [u8]);

    fn append_scalars<F: PrimeField>(&mut self, scalars: &[impl Borrow<F>]) {
        let mut buff = vec![0u8; F::MODULUS_BIT_SIZE as usize / 8];
        for scalar in scalars {
            scalar
                .borrow()
                .serialize_compressed(&mut buff)
                .expect("should never fail");
            self.append_bytes(&buff);
        }
    }

    fn challenge_scalar<F: PrimeField>(&mut self) -> F {
        // /2 because for challenge we dont need 256 bits we only need 128 bits
        let mut challenge = [0u8; 16];
        self.challenge_bytes(&mut challenge);
        F::from_le_bytes_mod_order(&challenge)
    }

    /// Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: PrimeField>(&mut self, count: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); count];
        for i in 1..count {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    fn append_points<A: AffineRepr>(&mut self, points: &[impl Borrow<A>]) {
        let mut buff =
            vec![0u8; <A::BaseField as Field>::BasePrimeField::MODULUS_BIT_SIZE as usize / 8];
        for point in points {
            let aff = point.borrow();
            if aff.is_zero() {
                // TODO: unclear how to handle this - jolt's doing 64 bytes but it's not if it's really useful
                self.append_bytes(&ZERO_POINT_SERIALIZED);
                continue;
            }
            let (x, y) = aff.xy().expect("point should be on curve");
            x.serialize_compressed(&mut buff)
                .expect("should never fail");
            self.append_bytes(&buff);
            y.serialize_compressed(&mut buff)
                .expect("should never fail");
            self.append_bytes(&buff);
        }
    }
}

pub trait AppendToTranscript {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript);
}
