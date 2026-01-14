use std::borrow::Borrow;

use ark_ec::{scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup};
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
        batch_poly_msm_gpu_g1_bn254(g1_powers, polys)
    } else if TypeId::of::<A>() == TypeId::of::<ark_bn254::G2Affine>() {
        batch_poly_msm_gpu_g2_bn254(g1_powers, polys)
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
pub fn batch_poly_msm_gpu_g1_bn254<'a, A: AffineRepr>(
    g1_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    use std::sync::Arc;

    use super::gpu_msm::{convert_g1_bases_to_gpu, convert_scalars_to_bigint, GPU_MSM_G1};

    let bases: &[ark_bn254::G1Affine] = unsafe { std::mem::transmute(g1_powers) };
    let bases_gpu = Arc::new(convert_g1_bases_to_gpu(bases));

    let results: Vec<A::Group> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.borrow().evals_ref();
            let scalars: &[ark_bn254::Fr] = unsafe { std::mem::transmute(coeffs) };
            let msm_size = scalars.len();
            let scalars_bigint = Arc::new(convert_scalars_to_bigint(&scalars[..msm_size]));

            let bases_slice = Arc::new(bases_gpu[..msm_size].to_vec());

            let result: ark_bn254::G1Projective = GPU_MSM_G1
                .lock()
                .unwrap()
                .msm_arc(bases_slice, scalars_bigint)
                .map_err(|e| anyhow::anyhow!("GPU MSM error: {e}"))?;

            let result_generic: A::Group = unsafe { std::mem::transmute_copy(&result) };
            Ok(result_generic)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(results)
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn batch_poly_msm_gpu_g2_bn254<'a, A: AffineRepr>(
    g2_powers: &[A],
    polys: &[impl Borrow<DensePolynomial<'a, A::ScalarField>>],
) -> anyhow::Result<Vec<A::Group>> {
    use std::sync::Arc;

    use super::gpu_msm::{convert_g2_bases_to_gpu, convert_scalars_to_bigint, GPU_MSM_G2};

    let bases: &[ark_bn254::G2Affine] = unsafe { std::mem::transmute(g2_powers) };
    let bases_gpu = Arc::new(convert_g2_bases_to_gpu(bases));

    let results: Vec<A::Group> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.borrow().evals_ref();
            let scalars: &[ark_bn254::Fr] = unsafe { std::mem::transmute(coeffs) };
            let msm_size = scalars.len();
            let scalars_bigint = Arc::new(convert_scalars_to_bigint(&scalars[..msm_size]));

            let bases_slice = Arc::new(bases_gpu[..msm_size].to_vec());

            let result: ark_bn254::G2Projective = GPU_MSM_G2
                .lock()
                .unwrap()
                .msm_arc(bases_slice, scalars_bigint)
                .map_err(|e| anyhow::anyhow!("GPU MSM error: {e}"))?;

            let result_generic: A::Group = unsafe { std::mem::transmute_copy(&result) };
            Ok(result_generic)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

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
