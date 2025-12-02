use super::{Expression, StructuralWitIn, WitIn};
use crate::sumcheck::{Fixed, Instance, WitnessId, monomial::Term};
use ark_ff::PrimeField;
use itertools::Itertools;

impl WitIn {
    pub fn assign<F: PrimeField>(&self, instance: &mut [F], value: F) {
        instance[self.id as usize] = value;
    }
}

impl StructuralWitIn {
    pub fn assign<F: PrimeField>(&self, instance: &mut [F], value: F) {
        instance[self.id as usize] = value;
    }
}

pub fn eval_by_expr_constant<F: PrimeField>(
    instance: &[F],
    challenges: &[F],
    expr: &Expression<F>,
) -> F {
    expr.evaluate_with_instance(
        &|_| unimplemented!(),
        &|_| unimplemented!(),
        &|_, _, _, _| unimplemented!(),
        &|i| instance[i.0],
        &|scalar| scalar,
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            challenge.pow(vec![pow as u64]) * scalar + offset
        },
        &|a, b| a + b,
        &|a, b| a * b,
        &|x, a, b| a * x + b,
    )
}

pub fn eval_by_expr<F: PrimeField>(
    witnesses: &[F],
    structural_witnesses: &[F],
    challenges: &[F],
    expr: &Expression<F>,
) -> F {
    eval_by_expr_with_fixed(&[], witnesses, structural_witnesses, challenges, expr)
}

/// Evaluates the expression using fixed values, witnesses, structural witnesses, and challenges.
/// We allow shorter fixed vectors, which are of the length 2^k and repeated cyclically. `fixed_len_minus_one` is the
/// length of the fixed vector minus one, which is used to wrap around the indices.
pub fn eval_by_expr_with_fixed<F: PrimeField>(
    fixed: &[F],
    witnesses: &[F],
    structural_witnesses: &[F],
    challenges: &[F],
    expr: &Expression<F>,
) -> F {
    expr.evaluate::<F>(
        &|f| fixed[f.0],
        &|witness_id| witnesses[witness_id as usize],
        &|witness_id, _, _, _| structural_witnesses[witness_id as usize],
        &|scalar| scalar,
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            challenge.pow(vec![pow as u64]) * scalar + offset
        },
        &|a, b| a + b,
        &|a, b| a * b,
        &|x, a, b| a * x + b,
    )
}

pub fn eval_by_expr_with_instance<F: PrimeField>(
    fixed: &[F],
    witnesses: &[F],
    structural_witnesses: &[F],
    instance: &[F],
    challenges: &[F],
    expr: &Expression<F>,
) -> F {
    expr.evaluate_with_instance::<F>(
        &|f| fixed[f.0],
        &|witness_id| witnesses[witness_id as usize],
        &|witness_id, _, _, _| structural_witnesses[witness_id as usize],
        &|i| instance[i.0],
        &|scalar| scalar,
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            challenge.pow(vec![pow as u64]) * scalar + offset
        },
        &|a, b| a + b,
        &|a, b| a * b,
        &|x, a, b| a * x + b,
    )
}

/// convert complex expression into monomial form to WitIn
/// orders WitIn ++ StructuralWitIn ++ Fixed
pub fn monomialize_expr_to_wit_terms<F: PrimeField>(
    expr: &Expression<F>,
    num_witin: WitnessId,
    num_structural_witin: WitnessId,
    num_fixed: WitnessId,
) -> Vec<Term<Expression<F>, Expression<F>>> {
    let witid_offset = 0 as WitnessId;
    let structural_witin_offset = witid_offset + num_witin;
    let fixed_offset = structural_witin_offset + num_structural_witin;
    let instance_offset = fixed_offset + num_fixed;

    let monomial_terms_expr = expr.get_monomial_terms();
    monomial_terms_expr
        .into_iter()
        .map(
            |Term {
                 scalar,
                 mut product,
             }| {
                product.iter_mut().for_each(|t| match t {
                    Expression::WitIn(_) => (),
                    Expression::StructuralWitIn(structural_wit_id, _, _, _) => {
                        *t = Expression::WitIn(structural_witin_offset + *structural_wit_id);
                    }
                    Expression::Fixed(Fixed(fixed_id)) => {
                        *t = Expression::WitIn(fixed_offset + (*fixed_id as u16));
                    }
                    Expression::Instance(Instance(instance_id)) => {
                        *t = Expression::WitIn(instance_offset + (*instance_id as u16));
                    }
                    e => panic!("unknown monomial terms {:?}", e),
                });
                Term { scalar, product }
            },
        )
        .collect_vec()
}
