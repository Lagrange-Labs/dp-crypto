use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};

pub fn random_challenge<F: PrimeField, R: RngCore>(rng: &mut R) -> F {
    new_from_u128(rng.r#gen::<u128>()).unwrap()
}
pub fn new_from_u128<F: PrimeField>(value: u128) -> Option<F> {
    let bg = F::BigInt::from(value);
    F::from_bigint(bg)
}