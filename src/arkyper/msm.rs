use std::borrow::Borrow;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM};
use ark_ff::{Field, PrimeField};
use ark_std::cfg_iter;

use crate::poly::dense::DensePolynomial;
pub fn poly_msm<A: AffineRepr>(g1_powers: &[A], poly: &impl Borrow<DensePolynomial<A::ScalarField>>) -> anyhow::Result<A::Group> {
    let mut r = batch_poly_msm(g1_powers, &[poly.borrow()])?;
    Ok(r.remove(0))
}

pub fn batch_poly_msm<A: AffineRepr>(g1_powers: &[A], polys: &[impl Borrow<DensePolynomial<A::ScalarField>>]) -> anyhow::Result<Vec<A::Group>> {
    let coeffs = polys.iter().map(|p| p.borrow().evals_ref()).collect::<Vec<_>>();
    let r = cfg_iter!(coeffs).map(|coeffs| {
        let msm_size = coeffs.len();
        // TODO: move to msm_bigint as they do in arkworks KZG
        // https://github.com/arkworks-rs/poly-commit/blob/master/poly-commit/src/kzg10/mod.rs#L171-L204
        <A::Group as VariableBaseMSM>::msm(&g1_powers[..msm_size], coeffs).map_err(|e| anyhow::anyhow!("MSM error: {e}"))
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(r)
}

pub fn msm<G: CurveGroup<ScalarField = F>, F: PrimeField>(g_powers: &[G::Affine], coeffs: &[F]) -> anyhow::Result<G> { 
    let r = <G as VariableBaseMSM>::msm(g_powers, coeffs).map_err(|e| anyhow::anyhow!("MSM error: {e}"))?;
    Ok(r)
}