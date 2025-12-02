use ark_ff::PrimeField;
use itertools::{Itertools, chain, iproduct};
use serde::{Deserialize, Serialize};

use super::Expression;
use Expression::*;
use std::{fmt::Display, iter::Sum};

impl<F: PrimeField> Expression<F> {
    pub fn get_monomial_terms(&self) -> Vec<Term<Expression<F>, Expression<F>>> {
        Self::combine(self.distribute())
            .into_iter()
            // filter coeff = 0 monimial terms
            .filter(|Term { scalar, .. }| match scalar {
                // filter term with scalar != zero
                Constant(scalar) => *scalar != F::ZERO,
                _ => true,
            })
            .collect_vec()
    }

    fn distribute(&self) -> Vec<Term<Expression<F>, Expression<F>>> {
        match self {
            // only contribute to scalar terms
            Constant(_) | Challenge(..) | InstanceScalar(_) => {
                vec![Term {
                    scalar: self.clone(),
                    product: vec![],
                }]
            }

            Fixed(_) | Instance(_) | WitIn(_) | StructuralWitIn(..) => {
                vec![Term {
                    scalar: Expression::ONE,
                    product: vec![self.clone()],
                }]
            }

            Sum(a, b) => chain!(a.distribute(), b.distribute()).collect(),

            Product(a, b) => iproduct!(a.distribute(), b.distribute())
                .map(|(a, b)| Term {
                    scalar: &a.scalar * &b.scalar,
                    product: chain!(&a.product, &b.product).cloned().collect(),
                })
                .collect(),

            ScaledSum(x, a, b) => chain!(
                b.distribute(),
                iproduct!(x.distribute(), a.distribute()).map(|(x, a)| Term {
                    scalar: &x.scalar * &a.scalar,
                    product: chain!(&x.product, &a.product).cloned().collect(),
                })
            )
            .collect(),
        }
    }

    fn combine(
        mut terms: Vec<Term<Expression<F>, Expression<F>>>,
    ) -> Vec<Term<Expression<F>, Expression<F>>> {
        for Term { product, .. } in &mut terms {
            product.sort();
        }
        terms
            .into_iter()
            .map(|Term { scalar, product }| (product, scalar))
            .into_group_map()
            .into_iter()
            .map(|(product, scalar)| Term {
                scalar: scalar.into_iter().sum(),
                product,
            })
            .collect()
    }
}

impl<F: PrimeField> Sum<Term<Expression<F>, Expression<F>>> for Expression<F> {
    fn sum<I: Iterator<Item = Term<Expression<F>, Expression<F>>>>(iter: I) -> Self {
        iter.map(|term| term.scalar * term.product.into_iter().product::<Expression<_>>())
            .sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Term<S, P> {
    pub scalar: S,
    pub product: Vec<P>,
}

impl<F: PrimeField> Display for Term<Expression<F>, Expression<F>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // join the product terms with " * "
        let product_str = self
            .product
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" * ");
        // format as: scalar * (a * b * c)
        write!(f, "{} * ({})", self.scalar, product_str)
    }
}

#[cfg(test)]
mod tests {
    use crate::sumcheck::expression::{Fixed as FixedS, utils::eval_by_expr_with_fixed};

    use super::*;
    use ark_ff::{AdditiveGroup, Field, UniformRand};
    use ark_std::rand::thread_rng;

    type F = ark_bn254::Fq;

    #[test]
    fn test_to_monomial_form() {
        use Expression::*;

        let eval = make_eval();

        let a = || Fixed(FixedS(0));
        let b = || Fixed(FixedS(1));
        let c = || Fixed(FixedS(2));
        let x1 = || WitIn(0);
        let x2 = || WitIn(1);
        let x3 = || WitIn(2);
        let x4 = || WitIn(3);
        let x5 = || WitIn(4);
        let x6 = || WitIn(5);
        let x7 = || WitIn(6);

        let n1 = || Constant(F::from(103u64));
        let n2 = || Constant(F::from(101u64));
        let m = || Constant(F::from(-599));
        let r = || Challenge(0, 1, F::ONE, F::ZERO);

        let test_exprs: &[Expression<F>] = &[
            a() * x1() * x2(),
            a(),
            x1(),
            n1(),
            r(),
            a() + b() + x1() + x2() + n1() + m() + r(),
            a() * x1() * n1() * r(),
            x1() * x2() * x3(),
            (x1() + x2() + a()) * b() * (x2() + x3()) + c(),
            (r() * x1() + n1() + x3()) * m() * x2(),
            (b() + x2() + m() * x3()) * (x1() + x2() + c()),
            a() * r() * x1(),
            x1() * (n1() * (x2() * x3() + x4() * x5())) + n2() * x2() * x4() + x1() * x6() * x7(),
        ];

        for factored in test_exprs {
            let monomials = factored
                .get_monomial_terms()
                .into_iter()
                .sum::<Expression<F>>();
            assert!(monomials.is_monomial_form());

            // Check that the two forms are equivalent (Schwartz-Zippel test).
            let factored = eval(factored);
            let monomials = eval(&monomials);
            assert_eq!(monomials, factored);
        }
    }

    /// Create an evaluator of expressions. Fixed, witness, and challenge values are pseudo-random.
    fn make_eval() -> impl Fn(&Expression<F>) -> F {
        // Create a deterministic RNG from a seed.
        let mut rng = thread_rng();
        let fixed = vec![F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng)];
        let witnesses = vec![
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
        ];
        let challenges = vec![F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng)];
        move |expr: &Expression<F>| {
            eval_by_expr_with_fixed(&fixed, &witnesses, &[], &challenges, expr)
        }
    }
}
