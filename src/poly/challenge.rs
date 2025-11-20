use ark_ff::PrimeField;
use ark_std::rand::{Rng, RngCore};

pub fn random_challenge<F: PrimeField, R: RngCore + Rng>(rng: &mut R) -> F
where
    <F as PrimeField>::BigInt: From<u128>,
{
    new_from_u128(rng.r#gen::<u128>()).unwrap()
}
pub fn new_from_u128<F: PrimeField>(value: u128) -> Option<F>
// XXX why do we need to add it again while it's already in the F::BigInt trait
where
    <F as PrimeField>::BigInt: From<u128>,
{
    let bg = F::BigInt::from(value);
    F::from_bigint(bg)
}
