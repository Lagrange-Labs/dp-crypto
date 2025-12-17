use std::borrow::Borrow;
use std::sync::Once;

use ark_bn254::{Fr as Bn254Fr, G1Affine as Bn254G1Affine};
use ark_ec::AffineRepr;
use ark_ff::BigInteger;
use ark_ff::PrimeField;
use blitzar::compute::init_backend;

use crate::poly::dense::DensePolynomial;

static INIT: Once = Once::new();

pub fn init_blitzar_backend() {
    INIT.call_once(|| {
        init_backend();
    });
}

pub fn bn254_blitzar_msm(
    points: &[Bn254G1Affine],
    scalars: &[Bn254Fr],
) -> anyhow::Result<Bn254G1Affine> {
    if points.len() != scalars.len() {
        anyhow::bail!("Points and scalars must have the same length");
    }

    if points.is_empty() {
        return Ok(Bn254G1Affine::zero());
    }

    // Initialize backend if not already done
    init_blitzar_backend();

    // Convert scalars to blitzar format
    let scalar_bytes: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| s.into_bigint().to_bytes_le().try_into().unwrap())
        .collect();
    let mut blitzar_commitments = vec![Default::default(); 1];

    blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_generators(
        &mut blitzar_commitments,
        &[(&scalar_bytes).into()],
        points,
    );

    Ok(blitzar_commitments[0])
}

pub fn bn254_poly_msm<'a>(
    g1_powers: &[Bn254G1Affine],
    poly: &impl Borrow<DensePolynomial<'a, Bn254Fr>>,
) -> anyhow::Result<Bn254G1Affine> {
    let coeffs = poly.borrow().evals_ref();
    bn254_blitzar_msm(&g1_powers[..coeffs.len()], coeffs)
}

pub fn bn254_batch_poly_msm<'a>(
    g1_powers: &[Bn254G1Affine],
    polys: &[impl Borrow<DensePolynomial<'a, Bn254Fr>>],
) -> anyhow::Result<Vec<Bn254G1Affine>> {
    if polys.is_empty() {
        return Ok(vec![]);
    }

    init_blitzar_backend();

    let all_scalar_bytes: Vec<Vec<[u8; 32]>> = polys
        .iter()
        .map(|poly| {
            let coeffs = poly.borrow().evals_ref();
            coeffs
                .iter()
                .map(|s| s.into_bigint().to_bytes_le().try_into().unwrap())
                .collect()
        })
        .collect();

    let scalar_refs: Vec<_> = all_scalar_bytes
        .iter()
        .map(|v| v.as_slice().into())
        .collect();

    let mut blitzar_commitments = vec![Default::default(); polys.len()];

    blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_generators(
        &mut blitzar_commitments,
        &scalar_refs,
        g1_powers,
    );

    Ok(blitzar_commitments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Projective;
    use ark_ec::CurveGroup;
    use ark_std::UniformRand;
    use ark_std::rand::thread_rng;

    #[test]
    fn test_blitzar_backend_init() {
        // Should not panic
        init_blitzar_backend();
        init_blitzar_backend(); // Should be safe to call multiple times
    }

    #[test]
    fn test_bn254_blitzar_msm_empty() {
        let result = bn254_blitzar_msm(&[], &[]).unwrap();
        assert!(result.is_zero());
    }

    #[test]
    fn test_bn254_blitzar_msm_single() {
        let mut rng = thread_rng();
        let point = Bn254G1Affine::rand(&mut rng);
        let scalar = Bn254Fr::rand(&mut rng);

        let result = bn254_blitzar_msm(&[point], &[scalar]).unwrap();

        // Compare with arkworks result
        let expected = (G1Projective::from(point) * scalar).into_affine();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bn254_blitzar_msm_multiple() {
        let mut rng = thread_rng();
        let n = 100;

        let points: Vec<Bn254G1Affine> = (0..n).map(|_| Bn254G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Bn254Fr> = (0..n).map(|_| Bn254Fr::rand(&mut rng)).collect();

        let result = bn254_blitzar_msm(&points, &scalars).unwrap();

        // Compare with arkworks result
        let expected = points
            .iter()
            .zip(scalars.iter())
            .map(|(p, s)| G1Projective::from(*p) * s)
            .sum::<G1Projective>()
            .into_affine();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bn254_poly_msm() {
        let mut rng = thread_rng();
        let n = 4; // 2^n coefficients
        let size = 1 << n;

        let points: Vec<Bn254G1Affine> = (0..size).map(|_| Bn254G1Affine::rand(&mut rng)).collect();

        let coeffs: Vec<Bn254Fr> = (0..size).map(|_| Bn254Fr::rand(&mut rng)).collect();

        let poly = DensePolynomial::new(coeffs.clone());

        let result = bn254_poly_msm(&points, &poly).unwrap();

        // Compare with direct MSM
        let expected = bn254_blitzar_msm(&points, &coeffs).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bn254_batch_poly_msm() {
        let mut rng = thread_rng();
        let n = 4;
        let size = 1 << n;
        let batch_size = 3;

        let points: Vec<Bn254G1Affine> = (0..size).map(|_| Bn254G1Affine::rand(&mut rng)).collect();

        let polys: Vec<DensePolynomial<Bn254Fr>> = (0..batch_size)
            .map(|_| {
                let coeffs: Vec<Bn254Fr> = (0..size).map(|_| Bn254Fr::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs.clone())
            })
            .collect();

        let results = bn254_batch_poly_msm(&points, &polys).unwrap();

        assert_eq!(results.len(), batch_size);

        // Verify each result
        for (i, result) in results.iter().enumerate() {
            let expected = bn254_poly_msm(&points, &polys[i]).unwrap();
            assert_eq!(*result, expected);
        }
    }
}
