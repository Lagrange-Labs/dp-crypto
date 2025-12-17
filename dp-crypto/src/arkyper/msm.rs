use std::borrow::Borrow;

use ark_ec::{AffineRepr, CurveGroup, scalar_mul::variable_base::VariableBaseMSM};
use ark_ff::PrimeField;
use ark_std::cfg_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::poly::dense::DensePolynomial;

use ark_bn254::{Fr as Bn254Fr, G1Affine as Bn254G1Affine, G1Projective as Bn254G1Projective};

// Generic implementations (fallback for all curves)

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
            // TODO: move to msm_bigint as they do in arkworks KZG
            // https://github.com/arkworks-rs/poly-commit/blob/master/poly-commit/src/kzg10/mod.rs#L171-L204
            <A::Group as VariableBaseMSM>::msm(&g1_powers[..msm_size], coeffs)
                .map_err(|e| anyhow::anyhow!("MSM error: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(r)
}

pub fn msm<G: CurveGroup<ScalarField = F>, F: PrimeField>(
    g_powers: &[G::Affine],
    coeffs: &[F],
) -> anyhow::Result<G> {
    let r = <G as VariableBaseMSM>::msm(g_powers, coeffs)
        .map_err(|e| anyhow::anyhow!("MSM error: {e}"))?;
    Ok(r)
}

// BN254 specialized implementations using blitzar when feature is enabled

pub fn poly_msm_bn254<'a>(
    g1_powers: &[Bn254G1Affine],
    poly: &impl Borrow<DensePolynomial<'a, Bn254Fr>>,
) -> anyhow::Result<Bn254G1Projective> {
    #[cfg(feature = "blitzar-msm")]
    {
        use crate::arkyper::blitzar_msm;
        let result = blitzar_msm::bn254_poly_msm(g1_powers, poly)?;
        Ok(result.into_group())
    }

    #[cfg(not(feature = "blitzar-msm"))]
    {
        poly_msm(g1_powers, poly)
    }
}

pub fn batch_poly_msm_bn254<'a>(
    g1_powers: &[Bn254G1Affine],
    polys: &[impl Borrow<DensePolynomial<'a, Bn254Fr>>],
) -> anyhow::Result<Vec<Bn254G1Projective>> {
    #[cfg(feature = "blitzar-msm")]
    {
        use crate::arkyper::blitzar_msm;
        let results = blitzar_msm::bn254_batch_poly_msm(g1_powers, polys)?;
        Ok(results.into_iter().map(|r| r.into_group()).collect())
    }

    #[cfg(not(feature = "blitzar-msm"))]
    {
        batch_poly_msm(g1_powers, polys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine};
    use ark_std::rand::thread_rng;
    use ark_std::{UniformRand, Zero};

    #[test]
    fn test_bn254_msm() {
        let mut rng = thread_rng();
        let n = 100;

        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        let result: Bn254G1Projective = msm(&points, &scalars).unwrap();

        assert!(!result.is_zero());
    }

    #[test]
    fn test_bn254_specialized_poly_msm() {
        let mut rng = thread_rng();
        let n = 4;
        let size = 1 << n;

        let points: Vec<G1Affine> = (0..size).map(|_| G1Affine::rand(&mut rng)).collect();
        let coeffs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DensePolynomial::new(coeffs.clone());

        // Test specialized version
        let result_specialized = poly_msm_bn254(&points, &poly).unwrap();

        // Test generic version
        let result_generic: Bn254G1Projective = poly_msm(&points, &poly).unwrap();

        assert_eq!(result_specialized, result_generic);
    }

    #[test]
    fn test_bn254_specialized_batch_poly_msm() {
        let mut rng = thread_rng();
        let n = 4;
        let size = 1 << n;
        let batch_size = 3;

        let points: Vec<G1Affine> = (0..size).map(|_| G1Affine::rand(&mut rng)).collect();
        let polys: Vec<DensePolynomial<Fr>> = (0..batch_size)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs)
            })
            .collect();

        // Test specialized version
        let results_specialized = batch_poly_msm_bn254(&points, &polys).unwrap();

        // Test generic version
        let results_generic: Vec<Bn254G1Projective> = batch_poly_msm(&points, &polys).unwrap();

        assert_eq!(results_specialized.len(), results_generic.len());
        for (s, g) in results_specialized.iter().zip(results_generic.iter()) {
            assert_eq!(s, g);
        }
    }
}

#[cfg(all(test, feature = "blitzar-msm"))]
mod blitzar_comparison_tests {
    use super::*;
    use crate::arkyper::blitzar_msm;
    use ark_bn254::{Fr, G1Affine};
    use ark_ec::CurveGroup;
    use ark_std::UniformRand;
    use ark_std::rand::thread_rng;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_blitzar_vs_arkworks_msm() {
        let mut rng = thread_rng();
        let n = 100;

        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // Compute using arkworks (generic implementation)
        let arkworks_result: Bn254G1Projective = msm(&points, &scalars).unwrap();

        // Compute using blitzar
        let blitzar_result_affine = blitzar_msm::bn254_blitzar_msm(&points, &scalars).unwrap();
        let blitzar_result = blitzar_result_affine.into_group();

        assert_eq!(
            arkworks_result, blitzar_result,
            "Blitzar and arkworks MSM results should match"
        );
    }

    #[test]
    fn test_blitzar_vs_arkworks_poly_msm() {
        let mut rng = thread_rng();
        let n = 4;
        let size = 1 << n;

        let points: Vec<G1Affine> = (0..size).map(|_| G1Affine::rand(&mut rng)).collect();
        let coeffs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = DensePolynomial::new(coeffs);

        // Compute using arkworks (generic implementation)
        let arkworks_result: Bn254G1Projective = poly_msm(&points, &poly).unwrap();

        // Compute using blitzar
        let blitzar_result_affine = blitzar_msm::bn254_poly_msm(&points, &poly).unwrap();
        let blitzar_result = blitzar_result_affine.into_group();

        assert_eq!(
            arkworks_result, blitzar_result,
            "Blitzar and arkworks polynomial MSM results should match"
        );
    }

    #[test]
    fn test_blitzar_vs_arkworks_batch_poly_msm() {
        let mut rng = thread_rng();
        let n = 4;
        let size = 1 << n;
        let batch_size = 3;

        let points: Vec<G1Affine> = (0..size).map(|_| G1Affine::rand(&mut rng)).collect();
        let polys: Vec<DensePolynomial<Fr>> = (0..batch_size)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs)
            })
            .collect();

        // Compute using arkworks (generic implementation)
        let arkworks_results: Vec<Bn254G1Projective> = batch_poly_msm(&points, &polys).unwrap();

        // Compute using blitzar
        let blitzar_results_affine = blitzar_msm::bn254_batch_poly_msm(&points, &polys).unwrap();
        let blitzar_results: Vec<Bn254G1Projective> = blitzar_results_affine
            .into_iter()
            .map(|r| r.into_group())
            .collect();

        assert_eq!(arkworks_results.len(), blitzar_results.len());
        for (i, (ark, blz)) in arkworks_results
            .iter()
            .zip(blitzar_results.iter())
            .enumerate()
        {
            assert_eq!(
                ark, blz,
                "Blitzar and arkworks batch polynomial MSM results should match at index {i}",
            );
        }
    }

    #[test]
    fn test_blitzar_vs_arkworks_msm_empty() {
        use ark_std::Zero;

        // Test edge case: empty inputs
        let points: Vec<G1Affine> = vec![];
        let scalars: Vec<Fr> = vec![];

        let arkworks_result: Bn254G1Projective = msm(&points, &scalars).unwrap();
        let blitzar_result_affine = blitzar_msm::bn254_blitzar_msm(&points, &scalars).unwrap();
        let blitzar_result = blitzar_result_affine.into_group();

        assert_eq!(arkworks_result, blitzar_result);
        assert!(arkworks_result.is_zero());
    }

    #[test]
    fn test_blitzar_vs_arkworks_msm_single() {
        // Test edge case: single point
        let mut rng = thread_rng();
        let points = vec![G1Affine::rand(&mut rng)];
        let scalars = vec![Fr::rand(&mut rng)];

        let arkworks_result: Bn254G1Projective = msm(&points, &scalars).unwrap();
        let blitzar_result_affine = blitzar_msm::bn254_blitzar_msm(&points, &scalars).unwrap();
        let blitzar_result = blitzar_result_affine.into_group();

        assert_eq!(arkworks_result, blitzar_result);
    }

    #[test]
    fn test_blitzar_vs_arkworks_msm_large() {
        // Test with larger input to stress GPU acceleration
        let mut rng = thread_rng();
        let n = 1000;

        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        let arkworks_result: Bn254G1Projective = msm(&points, &scalars).unwrap();
        let blitzar_result_affine = blitzar_msm::bn254_blitzar_msm(&points, &scalars).unwrap();
        let blitzar_result = blitzar_result_affine.into_group();

        assert_eq!(
            arkworks_result, blitzar_result,
            "Blitzar and arkworks should match even with large inputs"
        );
    }
}
