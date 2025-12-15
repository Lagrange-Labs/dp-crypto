#![deny(clippy::cargo)]
pub mod expression;
pub mod extrapolate;

use std::sync::Arc;

use ark_ff::{Field, PrimeField};
use ark_std::rand::Rng;
pub use expression::{utils::monomialize_expr_to_wit_terms, *};


use crate::poly::dense::DensePolynomial;
#[macro_use]
pub mod macros;
pub mod prover;
pub mod structs;
//pub mod mle;
pub mod util;
pub mod verifier;
pub mod virtual_poly;
pub mod virtual_polys;

pub type ArcMultilinearExtension<'a, F> = Arc<DensePolynomial<'a, F>>;

/// this is to avoid conflict implementation for Into of Vec<Vec<E::BaseField>>
pub trait IntoMLE<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mle(self) -> T;
}

impl<'a, F: Field> IntoMLE<DensePolynomial<'a, F>> for Vec<F> {
    fn into_mle(self) -> DensePolynomial<'a, F> {
        assert!(self.len().is_power_of_two(), "{}", self.len());
        DensePolynomial::new(self)
    }
}
pub trait IntoMLEs<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mles(self) -> Vec<T>;
}

impl<'a, F: Field> IntoMLEs<DensePolynomial<'a, F>> for Vec<Vec<F>> {
    fn into_mles(self) -> Vec<DensePolynomial<'a, F>> {
        self.into_iter().map(|v| v.into_mle()).collect()
    }
}

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<'b, R: Rng, F: PrimeField>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> (Vec<DensePolynomial<'b, F>>, F) {
    let start = entered_span!("sample random mle list");
    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::ZERO;

    for _ in 0..(1 << nv) {
        let mut product = F::ONE;

        for e in multiplicands.iter_mut() {
            let val = F::rand(&mut *rng);
            e.push(val);
            product *= val
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| DensePolynomial::new(x))
        .collect();

    exit_span!(start);
    (list, sum)
}

#[cfg(test)]
mod test;
