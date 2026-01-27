use std::borrow::Borrow;

use ark_ec::{AffineRepr, CurveGroup, scalar_mul::variable_base::VariableBaseMSM};
use ark_ff::PrimeField;
use ark_std::cfg_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::poly::dense::DensePolynomial;

pub fn poly_msm<'a, A: AffineRepr>(
    g1_powers: &[A],
    poly: &impl Borrow<DensePolynomial<'a, A::ScalarField>>,
) -> anyhow::Result<A::Group> {
    let mut r = batch_poly_msm(g1_powers, &[poly.borrow()])?;
    Ok(r.remove(0))
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub fn batch_poly_msm<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    batch_poly_msm_cpu(g1_powers, polys)
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn batch_poly_msm<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    use std::any::TypeId;

    if TypeId::of::<A>() == TypeId::of::<ark_bn254::G1Affine>() {
        batch_poly_msm_gpu_bn254(g1_powers, polys)
    } else {
        batch_poly_msm_cpu(g1_powers, polys)
    }
}

pub fn batch_poly_msm_cpu<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    let coeffs = polys
        .iter()
        .map(|p| p.borrow().evals_ref())
        .collect::<Vec<_>>();
    let r = cfg_iter!(coeffs)
        .map(|coeffs| {
            let msm_size = coeffs.len();
            <A::Group as VariableBaseMSM>::msm(&g1_powers[..msm_size], coeffs)
                .map_err(|e| anyhow::anyhow!("MSM error: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(r)
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn batch_poly_msm_gpu_bn254<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    use super::gpu_msm::GPU_MSM;

    if polys.is_empty() {
        return Ok(vec![]);
    }

    // Collect scalar slices and find max length
    let scalar_slices: Vec<&[ark_bn254::Fr]> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.borrow().evals_ref();
            // SAFETY: A::ScalarField is ark_bn254::Fr (checked at call site via TypeId)
            unsafe { std::mem::transmute::<&[A::ScalarField], &[ark_bn254::Fr]>(coeffs) }
        })
        .collect();

    let max_len = scalar_slices.iter().map(|s| s.len()).max().unwrap_or(0);
    let bases: &[ark_bn254::G1Affine] = unsafe { std::mem::transmute(g1_powers) };

    // Use batch_msm - single lock acquisition, bases uploaded once
    let results = GPU_MSM
        .lock()
        .unwrap()
        .batch_msm(&bases[..max_len], &scalar_slices)
        .map_err(|e| anyhow::anyhow!("GPU batch MSM error: {e}"))?;

    // Convert results to generic type
    let results_generic: Vec<A::Group> = results
        .into_iter()
        .map(|r| unsafe { std::mem::transmute_copy(&r) })
        .collect();

    Ok(results_generic)
}

pub fn msm<G: CurveGroup<ScalarField = F>, F: PrimeField>(
    g_powers: &[G::Affine],
    coeffs: &[F],
) -> anyhow::Result<G> {
    let r = <G as VariableBaseMSM>::msm(g_powers, coeffs)
        .map_err(|e| anyhow::anyhow!("MSM error: {e}"))?;
    Ok(r)
}
