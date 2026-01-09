use anyhow::ensure;
use ark_ff::{PrimeField, UniformRand};
use ark_std::rand::{Rng, thread_rng};
use either::Either;
use itertools::Itertools;

type Fq = ark_bn254::Fq;
type Fr = ark_bn254::Fr;
type T = Blake3Transcript;

use crate::{
    arkyper::transcript::{Transcript, blake3::Blake3Transcript},
    extrapolate::extrapolate_uni_poly,
    monomial::Term,
    poly::dense::DensePolynomial,
    structs::{IOPProverState, IOPVerifierState},
    sumcheck::{ArcMultilinearExtension, util::bit_decompose, virtual_poly::build_eq_x_r},
    util::{ceil_log2, max_usable_threads},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
    virtual_polys::VirtualPolynomials,
};

#[test]
fn test_eq_xr() {
    let mut rng = thread_rng();
    for nv in 4..10 {
        let r: Vec<_> = (0..nv).map(|_| Fq::rand(&mut rng)).collect();
        let eq_x_r = build_eq_x_r(r.as_ref());
        let eq_x_r2 = build_eq_x_r_for_test(r.as_ref());
        assert_eq!(eq_x_r, eq_x_r2);
    }
}

/// Naive method to build eq(x, r).
/// Only used for testing purpose.
// Evaluate
//      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
// over r, which is
//      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
fn build_eq_x_r_for_test<F: PrimeField>(r: &[F]) -> ArcMultilinearExtension<'_, F> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    // First, we build array for {1 - r_i}
    let one_minus_r: Vec<F> = r.iter().map(|ri| F::ONE - *ri).collect();

    let num_var = r.len();
    let mut eval = vec![];

    for i in 0..1 << num_var {
        let mut current_eval = F::ONE;
        let bit_sequence = bit_decompose(i, num_var);

        for (&bit, (ri, one_minus_ri)) in bit_sequence.iter().zip(r.iter().zip(one_minus_r.iter()))
        {
            current_eval *= if bit { *ri } else { *one_minus_ri };
        }
        eval.push(current_eval);
    }

    let mle = DensePolynomial::new(eval);

    mle.into()
}

// test polynomial mixed with different num_var
#[test]
fn test_sumcheck_with_different_degree() {
    let log_max_thread = ceil_log2(max_usable_threads());
    let nv = vec![1, 2, 3, 4];
    for num_threads in 1..log_max_thread {
        test_sumcheck_with_different_degree_helper::<Fq>(1 << num_threads, &nv, false).unwrap();
    }
}

#[test]
fn test_sumcheck_with_different_degree_padded_polys() {
    let log_max_thread = ceil_log2(max_usable_threads());
    let nv = vec![1, 2, 3, 4];
    for num_threads in 1..log_max_thread {
        test_sumcheck_with_different_degree_helper::<Fq>(1 << num_threads, &nv, true).unwrap();
    }
}

fn test_sumcheck_with_different_degree_helper<F: PrimeField>(
    num_threads: usize,
    nv: &[usize],
    pad_polynomials: bool,
) -> anyhow::Result<()> {
    let mut rng = thread_rng();
    let degree = 2;
    let num_multiplicands_range = (degree, degree + 1);
    let num_products = 1;
    let mut transcript = T::new(b"test");

    let max_num_variables = *nv.iter().max().unwrap();
    let (mut monimials, asserted_sum) = VirtualPolynomials::<F>::random_monimials(
        nv,
        num_multiplicands_range,
        num_products,
        &mut rng,
        pad_polynomials,
    );

    let poly = VirtualPolynomials::<F>::new_from_monomials(
        num_threads,
        max_num_variables,
        monimials
            .iter_mut()
            .map(|Term { scalar, product }| Term {
                scalar: *scalar,
                product: product.iter().map(Either::Left).collect_vec(),
            })
            .collect_vec(),
    );

    let (proof, _) = IOPProverState::<F>::prove(poly.as_view(), &mut transcript);
    let mut transcript = T::new(b"test");
    let subclaim = IOPVerifierState::<F>::verify(
        asserted_sum,
        &proof,
        &VPAuxInfo {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        },
        &mut transcript,
    );
    let r = &subclaim.point;
    assert_eq!(r.len(), max_num_variables);
    // r are right alignment
    assert!(
        poly.evaluate_slow(r)? == subclaim.expected_evaluation,
        "wrong subclaim"
    );

    // test in-place work
    let mut transcript = T::new(b"test");
    let (proof_mut, _) = IOPProverState::<F>::prove(poly, &mut transcript);
    assert_eq!(proof, proof_mut, "different proof");

    Ok(())
}

fn test_sumcheck<F: PrimeField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> anyhow::Result<()> {
    let mut rng = thread_rng();
    let mut transcript = T::new(b"test");

    let (poly, asserted_sum) =
        VirtualPolynomial::<F>::random(&[nv], num_multiplicands_range, num_products, &mut rng);
    let poly_info = poly.aux_info.clone();
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<F>::prove_parallel(poly.as_view(), &mut transcript);

    let mut transcript = T::new(b"test");
    let subclaim = IOPVerifierState::<F>::verify(asserted_sum, &proof, &poly_info, &mut transcript);
    ensure!(
        poly.evaluate(&subclaim.point)? == subclaim.expected_evaluation,
        "wrong subclaim"
    );
    Ok(())
}

fn test_sumcheck_internal<F: PrimeField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> anyhow::Result<()> {
    let mut rng = thread_rng();
    let (poly, asserted_sum) =
        VirtualPolynomial::<F>::random(&[nv], num_multiplicands_range, num_products, &mut rng);
    let (poly_info, num_variables) = (poly.aux_info.clone(), poly.aux_info.max_num_variables);
    #[allow(deprecated)]
    let mut prover_state = IOPProverState::prover_init_parallel(poly.as_view());
    let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
    let mut challenge = None;

    let mut transcript = T::new(b"test");

    transcript.append_bytes(b"initializing transcript for testing");

    for _ in 0..num_variables {
        let prover_message =
            IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge);

        challenge = Some(IOPVerifierState::verify_round_and_update_state(
            &mut verifier_state,
            &prover_message,
            &mut transcript,
        ));
    }
    // pushing the last challenge point to the state
    if let Some(p) = challenge {
        prover_state.push_challenges(vec![p]);
        // fix last challenge to collect final evaluation
        prover_state.fix_var(&p);
    };
    let subclaim = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum);
    assert!(
        poly.evaluate(&subclaim.point)? == subclaim.expected_evaluation,
        "wrong subclaim"
    );
    Ok(())
}

#[test]
fn test_trivial_polynomial() {
    test_trivial_polynomial_helper::<Fq>().unwrap();
    test_trivial_polynomial_helper::<Fr>().unwrap();
}

fn test_trivial_polynomial_helper<F: PrimeField>() -> anyhow::Result<()> {
    let nv = 1;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<F>(nv, num_multiplicands_range, num_products)?;
    test_sumcheck_internal::<F>(nv, num_multiplicands_range, num_products)
}

#[test]
fn test_normal_polynomial() {
    test_normal_polynomial_helper::<Fq>().unwrap();
    test_normal_polynomial_helper::<Fr>().unwrap();
}

fn test_normal_polynomial_helper<F: PrimeField>() -> anyhow::Result<()> {
    let nv = 12;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<F>(nv, num_multiplicands_range, num_products)?;
    test_sumcheck_internal::<F>(nv, num_multiplicands_range, num_products)
}

#[test]
fn test_extract_sum() {
    test_extract_sum_helper::<Fq>();
    test_extract_sum_helper::<Fr>();
}

fn test_extract_sum_helper<F: PrimeField>() {
    let mut rng = thread_rng();
    let mut transcript = T::new(b"test");
    let (poly, asserted_sum) = VirtualPolynomial::<F>::random(&[8], (2, 3), 3, &mut rng);
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<F>::prove_parallel(poly, &mut transcript);
    assert_eq!(proof.extract_sum(), asserted_sum);
}

struct PolynomialEvals<F>(Vec<F>);

impl<F: PrimeField> PolynomialEvals<F> {
    fn rand_coeffs<R: Rng>(degree: usize, rng: &mut R) -> Self {
        Self((0..=degree).map(|_| F::rand(&mut *rng)).collect())
    }

    fn evaluate(&self, p: &F) -> F {
        let mut powers_of_p = *p;
        let mut res = self.0[0];
        for &c in self.0.iter().skip(1) {
            res += powers_of_p * c;
            powers_of_p *= *p;
        }
        res
    }
}

#[test]
fn test_extrapolation() {
    fn run_extrapolation_test(degree: usize) {
        let mut prng = thread_rng();
        let poly = PolynomialEvals::rand_coeffs(degree, &mut prng);
        let evals = (0..=degree)
            .map(|i| poly.evaluate(&Fq::from(i as u64)))
            .collect::<Vec<_>>();
        let query = Fq::rand(&mut prng);
        assert_eq!(poly.evaluate(&query), extrapolate_uni_poly(&evals, query));
    }

    run_extrapolation_test(1);
    run_extrapolation_test(2);
    run_extrapolation_test(3);
    run_extrapolation_test(4);
}
