use ark_ff::PrimeField;
use ark_std::rand::{Rng, RngCore};

pub fn random_challenge<F: PrimeField, R: RngCore + Rng>(rng: &mut R) -> F {
    new_from_u128(rng.r#gen::<u128>()).unwrap()
}
pub fn new_from_u128<F: PrimeField>(value: u128) -> Option<F> {
    // unwrap safe since we know it fits inside the field
    // TODO: adds assert
    F::from_random_bytes(&value.to_be_bytes())
}
