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

pub fn batch_poly_msm<'a, A: AffineRepr>(
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

#[cfg(feature = "cuda")]
pub fn batch_poly_msm_gpu_bn254<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    use super::gpu_msm::{GPU_MSM, convert_bases_to_gpu, convert_scalars_to_bigint};

    let bases: &[ark_bn254::G1Affine] = unsafe { std::mem::transmute(g1_powers) };

    // Find the maximum polynomial length to avoid uploading unnecessary bases.
    let max_len = polys
        .iter()
        .map(|p| p.borrow().evals_ref().len())
        .max()
        .unwrap_or(0);
    let bases_gpu = std::sync::Arc::new(convert_bases_to_gpu(&bases[..max_len]));

    // Use batch_multiexp to upload bases to GPU only once.
    // All scalar sets must match bases length, so zero-pad shorter ones.
    let scalar_sets: Vec<Vec<_>> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.borrow().evals_ref();
            let scalars: &[ark_bn254::Fr] = unsafe { std::mem::transmute(coeffs) };
            let mut bigints = convert_scalars_to_bigint(scalars);
            bigints.resize(max_len, Default::default());
            bigints
        })
        .collect();

    let results: Vec<ark_bn254::G1Projective> = GPU_MSM
        .lock()
        .unwrap()
        .batch_msm(bases_gpu, &scalar_sets)
        .map_err(|e| anyhow::anyhow!("GPU batch MSM error: {e}"))?;

    // Transmute results to generic type
    let results: Vec<A::Group> = results
        .into_iter()
        .map(|r| unsafe { std::mem::transmute_copy(&r) })
        .collect();

    Ok(results)
}

pub fn msm<G: CurveGroup<ScalarField = F>, F: PrimeField>(
    g_powers: &[G::Affine],
    coeffs: &[F],
) -> anyhow::Result<G> {
    let r = <G as VariableBaseMSM>::msm(g_powers, coeffs)
        .map_err(|e| anyhow::anyhow!("MSM error: {e}"))?;
    Ok(r)
}
