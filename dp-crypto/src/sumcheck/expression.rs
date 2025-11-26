pub mod monomial;
pub mod utils;

use crate::{
    poly::dense::DensePolynomial,
    sumcheck::{
        ArcMultilinearExtension, monomial::Term, monomialize_expr_to_wit_terms,
        utils::eval_by_expr_constant,
    },
};
use ark_ff::PrimeField;
use itertools::{Itertools, chain, izip};
use num_bigint::BigUint;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    cmp::max,
    fmt::{Debug, Display},
    iter::{Product, Sum, successors},
    ops::{Add, AddAssign, Deref, Index, Mul, MulAssign, Neg, Shl, ShlAssign, Sub, SubAssign},
};

pub type WitnessId = u16;
pub type ChallengeId = u16;
pub const MIN_PAR_SIZE: usize = 64;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + DeserializeOwned")]
pub enum Expression<F: PrimeField> {
    /// WitIn(Id)
    WitIn(WitnessId),
    /// StructuralWitIn is similar with WitIn, but it is structured.
    /// These witnesses in StructuralWitIn allow succinct verification directly during the verification processing, rather than requiring a commitment.
    /// StructuralWitIn(Id, max_len, offset, multi_factor)
    StructuralWitIn(WitnessId, usize, u32, usize),
    /// This multi-linear polynomial is known at the setup/keygen phase.
    Fixed(Fixed),
    /// Public Values
    Instance(Instance),
    /// Public Values, with global id counter shared with `Instance`
    InstanceScalar(Instance),
    /// Constant poly
    Constant(F),
    /// This is the sum of two expressions
    Sum(Box<Expression<F>>, Box<Expression<F>>),
    /// This is the product of two expressions
    Product(Box<Expression<F>>, Box<Expression<F>>),
    /// ScaledSum(x, a, b) represents a * x + b
    /// where x is one of wit / fixed / instance, a and b are either constants or challenges
    ScaledSum(Box<Expression<F>>, Box<Expression<F>>, Box<Expression<F>>),
    /// Challenge(challenge_id, power, scalar, offset)
    Challenge(ChallengeId, usize, F, F),
}

impl<F: PrimeField> Debug for Expression<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::WitIn(id) => write!(f, "W[{}]", id),
            Expression::StructuralWitIn(id, _max_len, _offset, _multi_factor) => {
                write!(f, "S[{}]", id)
            }
            Expression::Fixed(fixed) => write!(f, "F[{}]", fixed.0),
            Expression::Instance(instance) => write!(f, "I[{}]", instance.0),
            Expression::InstanceScalar(instance) => write!(f, "Is[{}]", instance.0),
            Expression::Constant(c) => write!(f, "Const[{}]", c),
            Expression::Sum(a, b) => write!(f, "({} + {})", a, b),
            Expression::Product(a, b) => write!(f, "({} * {})", a, b),
            Expression::ScaledSum(x, a, b) => write!(f, "{} * {} + {}", x, a, b),
            Expression::Challenge(challenge_id, pow, scalar, offset) => {
                write!(
                    f,
                    "C({})^{} * {:?} + {:?}",
                    challenge_id,
                    pow,
                    <F as Into<BigUint>>::into(*scalar).to_u64_digits(),
                    <F as Into<BigUint>>::into(*offset).to_u64_digits(),
                )
            }
        }
    }
}

/// this is used as finite state machine state
/// for differentiate an expression is in monomial form or not
enum MonomialState {
    SumTerm,
    ProductTerm,
}

impl<F: PrimeField> Expression<F> {
    pub const ZERO: Expression<F> = Expression::Constant(F::ZERO);
    pub const ONE: Expression<F> = Expression::Constant(F::ONE);

    pub fn id(&self) -> usize {
        match self {
            Expression::Fixed(Fixed(id)) => *id,
            Expression::WitIn(id) => *id as usize,
            Expression::StructuralWitIn(id, ..) => *id as usize,
            Expression::Instance(Instance(id)) => *id,
            Expression::InstanceScalar(Instance(id)) => *id,
            Expression::Constant(_) => unimplemented!(),
            Expression::Sum(..) => unimplemented!(),
            Expression::Product(..) => unimplemented!(),
            Expression::ScaledSum(..) => unimplemented!(),
            Expression::Challenge(id, _, _, _) => *id as usize,
        }
    }

    pub fn degree(&self) -> usize {
        match self {
            Expression::Fixed(_) => 1,
            Expression::WitIn(_) => 1,
            Expression::StructuralWitIn(..) => 1,
            Expression::Instance(_) => 1,
            Expression::InstanceScalar(_) => 0,
            Expression::Constant(_) => 0,
            Expression::Sum(a_expr, b_expr) => max(a_expr.degree(), b_expr.degree()),
            Expression::Product(a_expr, b_expr) => a_expr.degree() + b_expr.degree(),
            Expression::ScaledSum(x, _, _) => x.degree(),
            Expression::Challenge(_, _, _, _) => 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T>(
        &self,
        fixed_in: &impl Fn(&Fixed) -> T,
        wit_in: &impl Fn(WitnessId) -> T, // witin id
        structural_wit_in: &impl Fn(WitnessId, usize, u32, usize) -> T,
        constant: &impl Fn(F) -> T,
        challenge: &impl Fn(ChallengeId, usize, F, F) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, T, T) -> T,
    ) -> T {
        self.evaluate_with_instance(
            fixed_in,
            wit_in,
            structural_wit_in,
            &|_| unreachable!(),
            constant,
            challenge,
            sum,
            product,
            scaled,
        )
    }

    pub fn evaluate_constant<T>(
        &self,
        constant: &impl Fn(F) -> T,
        challenge: &impl Fn(ChallengeId, usize, F, F) -> T,
    ) -> T {
        match self {
            Expression::Constant(either) => constant(*either),
            Expression::Challenge(challenge_id, pow, scalar, offset) => {
                challenge(*challenge_id, *pow, *scalar, *offset)
            }
            _ => unimplemented!("unsupported"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_instance<T>(
        &self,
        fixed_in: &impl Fn(&Fixed) -> T,
        wit_in: &impl Fn(WitnessId) -> T, // witin id
        structural_wit_in: &impl Fn(WitnessId, usize, u32, usize) -> T,
        instance: &impl Fn(Instance) -> T,
        constant: &impl Fn(F) -> T,
        challenge: &impl Fn(ChallengeId, usize, F, F) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, T, T) -> T,
    ) -> T {
        match self {
            Expression::Fixed(f) => fixed_in(f),
            Expression::WitIn(witness_id) => wit_in(*witness_id),
            Expression::StructuralWitIn(witness_id, max_len, offset, multi_factor) => {
                structural_wit_in(*witness_id, *max_len, *offset, *multi_factor)
            }
            Expression::Instance(i) => instance(*i),
            Expression::InstanceScalar(i) => instance(*i),
            Expression::Constant(scalar) => constant(*scalar),
            Expression::Sum(a, b) => {
                let a = a.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                sum(a, b)
            }
            Expression::Product(a, b) => {
                let a = a.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                product(a, b)
            }
            Expression::ScaledSum(x, a, b) => {
                let x = x.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                let a = a.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate_with_instance(
                    fixed_in,
                    wit_in,
                    structural_wit_in,
                    instance,
                    constant,
                    challenge,
                    sum,
                    product,
                    scaled,
                );
                scaled(x, a, b)
            }
            Expression::Challenge(challenge_id, pow, scalar, offset) => {
                challenge(*challenge_id, *pow, *scalar, *offset)
            }
        }
    }

    pub fn is_monomial_form(&self) -> bool {
        Self::is_monomial_form_inner(MonomialState::SumTerm, self)
    }

    pub fn get_monomial_form(&self) -> Self {
        self.get_monomial_terms().into_iter().sum()
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Expression::Constant(_))
    }

    pub fn is_linear(&self) -> bool {
        self.degree() <= 1
    }

    fn is_zero_expr(expr: &Expression<F>) -> bool {
        match expr {
            Expression::Fixed(_) => false,
            Expression::WitIn(_) => false,
            Expression::StructuralWitIn(..) => false,
            Expression::Instance(_) => false,
            Expression::InstanceScalar(_) => false,
            Expression::Constant(c) => *c == F::ZERO,
            Expression::Sum(a, b) => Self::is_zero_expr(a) && Self::is_zero_expr(b),
            Expression::Product(a, b) => Self::is_zero_expr(a) || Self::is_zero_expr(b),
            Expression::ScaledSum(x, a, b) => {
                (Self::is_zero_expr(x) || Self::is_zero_expr(a)) && Self::is_zero_expr(b)
            }
            Expression::Challenge(_, _, scalar, offset) => *scalar == F::ZERO && *offset == F::ZERO,
        }
    }

    fn is_monomial_form_inner(s: MonomialState, expr: &Expression<F>) -> bool {
        match (expr, s) {
            (
                Expression::Fixed(_)
                | Expression::WitIn(_)
                | Expression::StructuralWitIn(..)
                | Expression::Challenge(..)
                | Expression::Constant(_)
                | Expression::Instance(_)
                | Expression::InstanceScalar(_),
                _,
            ) => true,
            (Expression::Sum(a, b), MonomialState::SumTerm) => {
                Self::is_monomial_form_inner(MonomialState::SumTerm, a)
                    && Self::is_monomial_form_inner(MonomialState::SumTerm, b)
            }
            (Expression::Sum(_, _), MonomialState::ProductTerm) => false,
            (Expression::Product(a, b), MonomialState::SumTerm) => {
                Self::is_monomial_form_inner(MonomialState::ProductTerm, a)
                    && Self::is_monomial_form_inner(MonomialState::ProductTerm, b)
            }
            (Expression::Product(a, b), MonomialState::ProductTerm) => {
                Self::is_monomial_form_inner(MonomialState::ProductTerm, a)
                    && Self::is_monomial_form_inner(MonomialState::ProductTerm, b)
            }
            (Expression::ScaledSum(_, _, _), MonomialState::SumTerm) => true,
            (Expression::ScaledSum(x, a, b), MonomialState::ProductTerm) => {
                Self::is_zero_expr(x) || Self::is_zero_expr(a) || Self::is_zero_expr(b)
            }
        }
    }

    /// recursively transforms an expression tree by allowing custom handlers for each leaf variant.
    /// this allows rewriting any part of the tree (e.g., replacing `Fixed` with `WitIn`, etc.).
    /// each closure corresponds to the rewrite a specific leaf node.
    #[allow(clippy::too_many_arguments)]
    pub fn transform_all<TF, TW, TS, TI, TIS, TC, CH>(
        &self,
        fixed_fn: &TF,
        wit_fn: &TW,
        struct_wit_fn: &TS,
        instance_fn: &TI,
        instance_scalar_fn: &TIS,
        constant_fn: &TC,
        challenge_fn: &CH,
    ) -> Expression<F>
    where
        TF: Fn(&Fixed) -> Expression<F>,
        TW: Fn(WitnessId) -> Expression<F>,
        TS: Fn(WitnessId, usize, u32, usize) -> Expression<F>,
        TI: Fn(Instance) -> Expression<F>,
        TIS: Fn(Instance) -> Expression<F>,
        TC: Fn(F) -> Expression<F>,
        CH: Fn(ChallengeId, usize, F, F) -> Expression<F>,
    {
        match self {
            Expression::WitIn(id) => wit_fn(*id),
            Expression::StructuralWitIn(id, len, offset, multi_factor) => {
                struct_wit_fn(*id, *len, *offset, *multi_factor)
            }
            Expression::Fixed(f) => fixed_fn(f),
            Expression::Instance(i) => instance_fn(*i),
            Expression::InstanceScalar(i) => instance_scalar_fn(*i),
            Expression::Constant(c) => constant_fn(*c),
            Expression::Challenge(id, pow, scalar, offset) => {
                challenge_fn(*id, *pow, *scalar, *offset)
            }
            Expression::Sum(a, b) => Expression::Sum(
                Box::new(a.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
                Box::new(b.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
            ),
            Expression::Product(a, b) => Expression::Product(
                Box::new(a.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
                Box::new(b.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
            ),
            Expression::ScaledSum(x, a, b) => Expression::ScaledSum(
                Box::new(x.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
                Box::new(a.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
                Box::new(b.transform_all(
                    fixed_fn,
                    wit_fn,
                    struct_wit_fn,
                    instance_fn,
                    instance_scalar_fn,
                    constant_fn,
                    challenge_fn,
                )),
            ),
        }
    }
}

impl<F: PrimeField> Neg for Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        match self {
            Expression::Fixed(_)
            | Expression::WitIn(_)
            | Expression::StructuralWitIn(..)
            | Expression::Instance(_) => Expression::ScaledSum(
                Box::new(self),
                Box::new(Expression::Constant(-F::ONE)),
                Box::new(Expression::Constant(F::ZERO)),
            ),

            Expression::Constant(c1) => Expression::Constant(-c1),
            Expression::Sum(a, b) => Expression::Sum(-a, -b),
            Expression::Product(a, b) => Expression::Product(-a, b.clone()),
            Expression::ScaledSum(x, a, b) => Expression::ScaledSum(x, -a, -b),
            Expression::Challenge(challenge_id, pow, scalar, offset) => {
                Expression::Challenge(challenge_id, pow, scalar.neg(), offset.neg())
            }
            Expression::InstanceScalar(_) => {
                unimplemented!("figure out how to support InstanceScalar")
            }
        }
    }
}

impl<F: PrimeField> Neg for &Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}

impl<F: PrimeField> Neg for Box<Expression<F>> {
    type Output = Box<Expression<F>>;
    fn neg(self) -> Self::Output {
        self.deref().clone().neg().into()
    }
}

impl<F: PrimeField> Neg for &Box<Expression<F>> {
    type Output = Box<Expression<F>>;
    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}

impl<F: PrimeField> Add for Expression<F> {
    type Output = Expression<F>;
    fn add(self, rhs: Expression<F>) -> Expression<F> {
        match (&self, &rhs) {
            // constant + witness
            // constant + fixed
            // constant + instance
            (Expression::WitIn(_), Expression::Constant(_))
            | (Expression::Fixed(_), Expression::Constant(_))
            | (Expression::Instance(_), Expression::Constant(_)) => Expression::ScaledSum(
                Box::new(self),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(rhs),
            ),
            (Expression::Constant(_), Expression::WitIn(_))
            | (Expression::Constant(_), Expression::Fixed(_))
            | (Expression::Constant(_), Expression::Instance(_)) => Expression::ScaledSum(
                Box::new(rhs),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(self),
            ),
            // challenge + witness
            // challenge + fixed
            // challenge + instance
            (Expression::WitIn(_), Expression::Challenge(..))
            | (Expression::Fixed(_), Expression::Challenge(..))
            | (Expression::Instance(_), Expression::Challenge(..)) => Expression::ScaledSum(
                Box::new(self),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(rhs),
            ),
            (Expression::Challenge(..), Expression::WitIn(_))
            | (Expression::Challenge(..), Expression::Fixed(_))
            | (Expression::Challenge(..), Expression::Instance(_)) => Expression::ScaledSum(
                Box::new(rhs),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(self),
            ),
            // constant + challenge
            (
                Expression::Constant(c1),
                Expression::Challenge(challenge_id, pow, scalar, offset),
            )
            | (
                Expression::Challenge(challenge_id, pow, scalar, offset),
                Expression::Constant(c1),
            ) => Expression::Challenge(*challenge_id, *pow, *scalar, *offset + c1),

            // challenge + challenge
            (
                Expression::Challenge(challenge_id1, pow1, scalar1, offset1),
                Expression::Challenge(challenge_id2, pow2, scalar2, offset2),
            ) => {
                if challenge_id1 == challenge_id2 && pow1 == pow2 {
                    Expression::Challenge(
                        *challenge_id1,
                        *pow1,
                        *scalar1 + *scalar2,
                        *offset1 + *offset2,
                    )
                } else {
                    Expression::Sum(Box::new(self), Box::new(rhs))
                }
            }

            // constant + constant
            (Expression::Constant(c1), Expression::Constant(c2)) => Expression::Constant(*c1 + *c2),

            // constant + scaled sum
            (c1 @ Expression::Constant(_), Expression::ScaledSum(x, a, b))
            | (Expression::ScaledSum(x, a, b), c1 @ Expression::Constant(_)) => {
                Expression::ScaledSum(x.clone(), a.clone(), Box::new(b.deref() + c1))
            }

            _ => Expression::Sum(Box::new(self), Box::new(rhs)),
        }
    }
}

macro_rules! binop_assign_instances {
    ($op_assign: ident, $fun_assign: ident, $op: ident, $fun: ident) => {
        impl<F: PrimeField, Rhs> $op_assign<Rhs> for Expression<F>
        where
            Expression<F>: $op<Rhs, Output = Expression<F>>,
        {
            fn $fun_assign(&mut self, rhs: Rhs) {
                // TODO: consider in-place?
                *self = self.clone().$fun(rhs);
            }
        }
    };
}

binop_assign_instances!(AddAssign, add_assign, Add, add);
binop_assign_instances!(SubAssign, sub_assign, Sub, sub);
binop_assign_instances!(MulAssign, mul_assign, Mul, mul);

impl<F: PrimeField> Shl<usize> for Expression<F> {
    type Output = Expression<F>;
    fn shl(self, rhs: usize) -> Expression<F> {
        self * (1_usize << rhs)
    }
}

impl<F: PrimeField> Shl<usize> for &Expression<F> {
    type Output = Expression<F>;
    fn shl(self, rhs: usize) -> Expression<F> {
        self.clone() << rhs
    }
}

impl<F: PrimeField> Shl<usize> for &mut Expression<F> {
    type Output = Expression<F>;
    fn shl(self, rhs: usize) -> Expression<F> {
        self.clone() << rhs
    }
}

impl<F: PrimeField> ShlAssign<usize> for Expression<F> {
    fn shl_assign(&mut self, rhs: usize) {
        *self = self.clone() << rhs;
    }
}

impl<F: PrimeField> Sum for Expression<F> {
    fn sum<I: Iterator<Item = Expression<F>>>(iter: I) -> Expression<F> {
        iter.fold(Expression::ZERO, |acc, x| acc + x)
    }
}

impl<F: PrimeField> Product for Expression<F> {
    fn product<I: Iterator<Item = Expression<F>>>(iter: I) -> Self {
        iter.fold(Expression::ONE, |acc, x| acc * x)
    }
}

impl<F: PrimeField> Sub for Expression<F> {
    type Output = Expression<F>;
    fn sub(self, rhs: Expression<F>) -> Expression<F> {
        match (&self, &rhs) {
            // witness - constant
            // fixed - constant
            // instance - constant
            (Expression::WitIn(_), Expression::Constant(_))
            | (Expression::Fixed(_), Expression::Constant(_))
            | (Expression::Instance(_), Expression::Constant(_)) => Expression::ScaledSum(
                Box::new(self),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(rhs.neg()),
            ),

            // constant - witness
            // constant - fixed
            // constant - instance
            (Expression::Constant(_), Expression::WitIn(_))
            | (Expression::Constant(_), Expression::Fixed(_))
            | (Expression::Constant(_), Expression::Instance(_)) => Expression::ScaledSum(
                Box::new(rhs),
                Box::new(Expression::Constant(F::ONE.neg())),
                Box::new(self),
            ),

            // witness - challenge
            // fixed - challenge
            // instance - challenge
            (Expression::WitIn(_), Expression::Challenge(..))
            | (Expression::Fixed(_), Expression::Challenge(..))
            | (Expression::Instance(_), Expression::Challenge(..)) => Expression::ScaledSum(
                Box::new(self),
                Box::new(Expression::Constant(F::ONE)),
                Box::new(rhs.neg()),
            ),

            // challenge - witness
            // challenge - fixed
            // challenge - instance
            (Expression::Challenge(..), Expression::WitIn(_))
            | (Expression::Challenge(..), Expression::Fixed(_))
            | (Expression::Challenge(..), Expression::Instance(_)) => Expression::ScaledSum(
                Box::new(rhs),
                Box::new(Expression::Constant(F::ONE.neg())),
                Box::new(self),
            ),

            // constant - challenge
            (
                Expression::Constant(c1),
                Expression::Challenge(challenge_id, pow, scalar, offset),
            ) => Expression::Challenge(*challenge_id, *pow, *scalar, offset.neg() + c1),

            // challenge - constant
            (
                Expression::Challenge(challenge_id, pow, scalar, offset),
                Expression::Constant(c1),
            ) => Expression::Challenge(*challenge_id, *pow, *scalar, *offset - c1),

            // challenge - challenge
            (
                Expression::Challenge(challenge_id1, pow1, scalar1, offset1),
                Expression::Challenge(challenge_id2, pow2, scalar2, offset2),
            ) => {
                if challenge_id1 == challenge_id2 && pow1 == pow2 {
                    Expression::Challenge(
                        *challenge_id1,
                        *pow1,
                        *scalar1 - *scalar2,
                        *offset1 - *offset2,
                    )
                } else {
                    Expression::Sum(Box::new(self), Box::new(-rhs))
                }
            }

            // constant - constant
            (Expression::Constant(c1), Expression::Constant(c2)) => Expression::Constant(*c1 - *c2),

            // constant - scalesum
            (c1 @ Expression::Constant(_), Expression::ScaledSum(x, a, b)) => {
                Expression::ScaledSum(x.clone(), -a, Box::new(c1 - b.deref()))
            }

            // scalesum - constant
            (Expression::ScaledSum(x, a, b), c1 @ Expression::Constant(_)) => {
                Expression::ScaledSum(x.clone(), a.clone(), Box::new(b.deref() - c1))
            }

            // challenge - scalesum
            (c1 @ Expression::Challenge(..), Expression::ScaledSum(x, a, b)) => {
                Expression::ScaledSum(x.clone(), -a, Box::new(c1 - b.deref()))
            }

            // scalesum - challenge
            (Expression::ScaledSum(x, a, b), c1 @ Expression::Challenge(..)) => {
                Expression::ScaledSum(x.clone(), a.clone(), Box::new(b.deref() - c1))
            }

            _ => Expression::Sum(Box::new(self), Box::new(-rhs)),
        }
    }
}

/// Instances for binary operations that mix Expression and &Expression
macro_rules! ref_binop_instances {
    ($op: ident, $fun: ident) => {
        impl<F: PrimeField> $op<&Expression<F>> for Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: &Expression<F>) -> Expression<F> {
                self.$fun(rhs.clone())
            }
        }

        impl<F: PrimeField> $op<Expression<F>> for &Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: Expression<F>) -> Expression<F> {
                self.clone().$fun(rhs)
            }
        }

        impl<F: PrimeField> $op<&Expression<F>> for &Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: &Expression<F>) -> Expression<F> {
                self.clone().$fun(rhs.clone())
            }
        }

        // for mutable references
        impl<F: PrimeField> $op<&mut Expression<F>> for Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: &mut Expression<F>) -> Expression<F> {
                self.$fun(rhs.clone())
            }
        }

        impl<F: PrimeField> $op<Expression<F>> for &mut Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: Expression<F>) -> Expression<F> {
                self.clone().$fun(rhs)
            }
        }

        impl<F: PrimeField> $op<&mut Expression<F>> for &mut Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: &mut Expression<F>) -> Expression<F> {
                self.clone().$fun(rhs.clone())
            }
        }
    };
}
ref_binop_instances!(Add, add);
ref_binop_instances!(Sub, sub);
ref_binop_instances!(Mul, mul);

macro_rules! mixed_binop_instances {
    ($op: ident, $fun: ident, ($($t:ty),*)) => {
        $(impl<F: PrimeField> $op<Expression<F>> for $t {
            type Output = Expression<F>;

            fn $fun(self, rhs: Expression<F>) -> Expression<F> {
                Expression::<F>::from(self).$fun(rhs)
            }
        }

        impl<F: PrimeField> $op<$t> for Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: $t) -> Expression<F> {
                self.$fun(Expression::<F>::from(rhs))
            }
        }

        impl<F: PrimeField> $op<&Expression<F>> for $t {
            type Output = Expression<F>;

            fn $fun(self, rhs: &Expression<F>) -> Expression<F> {
                Expression::<F>::from(self).$fun(rhs)
            }
        }

        impl<F: PrimeField> $op<$t> for &Expression<F> {
            type Output = Expression<F>;

            fn $fun(self, rhs: $t) -> Expression<F> {
                self.$fun(Expression::<F>::from(rhs))
            }
        }
    )*
    };
}

mixed_binop_instances!(
    Add,
    add,
    (u8, u16, u32, u64, usize, i8, i16, i32, i64, isize)
);
mixed_binop_instances!(
    Sub,
    sub,
    (u8, u16, u32, u64, usize, i8, i16, i32, i64, isize)
);
mixed_binop_instances!(
    Mul,
    mul,
    (u8, u16, u32, u64, usize, i8, i16, i32, i64, isize)
);

impl<F: PrimeField> Mul for Expression<F> {
    type Output = Expression<F>;
    fn mul(self, rhs: Expression<F>) -> Expression<F> {
        match (&self, &rhs) {
            // constant * witin
            // constant * fixed
            (c @ Expression::Constant(_), w @ Expression::WitIn(..))
            | (w @ Expression::WitIn(..), c @ Expression::Constant(_))
            | (c @ Expression::Constant(_), w @ Expression::Fixed(..))
            | (w @ Expression::Fixed(..), c @ Expression::Constant(_)) => Expression::ScaledSum(
                Box::new(w.clone()),
                Box::new(c.clone()),
                Box::new(Expression::Constant(F::ZERO)),
            ),
            // challenge * witin
            // challenge * fixed
            (c @ Expression::Challenge(..), w @ Expression::WitIn(..))
            | (w @ Expression::WitIn(..), c @ Expression::Challenge(..))
            | (c @ Expression::Challenge(..), w @ Expression::Fixed(..))
            | (w @ Expression::Fixed(..), c @ Expression::Challenge(..)) => Expression::ScaledSum(
                Box::new(w.clone()),
                Box::new(c.clone()),
                Box::new(Expression::Constant(F::ZERO)),
            ),
            // instance * witin
            // instance * fixed
            (c @ Expression::InstanceScalar(..), w @ Expression::WitIn(..))
            | (w @ Expression::WitIn(..), c @ Expression::InstanceScalar(..))
            | (c @ Expression::InstanceScalar(..), w @ Expression::Fixed(..))
            | (w @ Expression::Fixed(..), c @ Expression::InstanceScalar(..)) => {
                Expression::ScaledSum(
                    Box::new(w.clone()),
                    Box::new(c.clone()),
                    Box::new(Expression::Constant(F::ZERO)),
                )
            }
            // constant * challenge
            (
                Expression::Constant(c1),
                Expression::Challenge(challenge_id, pow, scalar, offset),
            )
            | (
                Expression::Challenge(challenge_id, pow, scalar, offset),
                Expression::Constant(c1),
            ) => Expression::Challenge(*challenge_id, *pow, *scalar * c1, *offset * c1),
            // challenge * challenge
            (
                Expression::Challenge(challenge_id1, pow1, s1, offset1),
                Expression::Challenge(challenge_id2, pow2, s2, offset2),
            ) => {
                if challenge_id1 == challenge_id2 {
                    // (s1 * s2 * c1^(pow1 + pow2) + offset2 * s1 * c1^(pow1) + offset1 * s2 * c2^(pow2))
                    // + offset1 * offset2

                    // (s1 * s2 * c1^(pow1 + pow2) + offset1 * offset2
                    let mut result = Expression::Challenge(
                        *challenge_id1,
                        pow1 + pow2,
                        *s1 * *s2,
                        *offset1 * *offset2,
                    );

                    // offset2 * s1 * c1^(pow1)
                    if *s1 != F::ZERO && *offset2 != F::ZERO {
                        result = Expression::Sum(
                            Box::new(result),
                            Box::new(Expression::Challenge(
                                *challenge_id1,
                                *pow1,
                                *offset2 * *s1,
                                F::ZERO,
                            )),
                        );
                    }

                    // offset1 * s2 * c2^(pow2))
                    if *s2 != F::ZERO && *offset1 != F::ZERO {
                        result = Expression::Sum(
                            Box::new(result),
                            Box::new(Expression::Challenge(
                                *challenge_id1,
                                *pow2,
                                *offset1 * *s2,
                                F::ZERO,
                            )),
                        );
                    }

                    result
                } else {
                    Expression::Product(Box::new(self), Box::new(rhs))
                }
            }

            // constant * constant
            (Expression::Constant(c1), Expression::Constant(c2)) => Expression::Constant(*c1 * *c2),
            // scaledsum * constant
            (Expression::ScaledSum(x, a, b), c2 @ Expression::Constant(_))
            | (c2 @ Expression::Constant(_), Expression::ScaledSum(x, a, b)) => {
                Expression::ScaledSum(
                    x.clone(),
                    Box::new(a.deref() * c2),
                    Box::new(b.deref() * c2),
                )
            }
            // scaled * challenge => scaled
            (Expression::ScaledSum(x, a, b), c2 @ Expression::Challenge(..))
            | (c2 @ Expression::Challenge(..), Expression::ScaledSum(x, a, b)) => {
                Expression::ScaledSum(
                    x.clone(),
                    Box::new(a.deref() * c2),
                    Box::new(b.deref() * c2),
                )
            }
            _ => Expression::Product(Box::new(self), Box::new(rhs)),
        }
    }
}

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct WitIn {
    pub id: WitnessId,
}

#[derive(Clone, Debug, Copy, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct StructuralWitIn {
    pub id: WitnessId,
    pub max_len: usize,
    pub offset: u32,
    pub multi_factor: usize,
    pub descending: bool,
}

#[derive(
    Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(C)]
pub struct Fixed(pub usize);

#[derive(
    Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(C)]
pub struct Instance(pub usize);

pub trait ToExpr<F: PrimeField> {
    type Output;
    fn expr(&self) -> Self::Output;
}

impl<F: PrimeField> ToExpr<F> for WitIn {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::WitIn(self.id)
    }
}

impl<F: PrimeField> ToExpr<F> for &WitIn {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::WitIn(self.id)
    }
}

impl<F: PrimeField> ToExpr<F> for StructuralWitIn {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::StructuralWitIn(self.id, self.max_len, self.offset, self.multi_factor)
    }
}

impl<F: PrimeField> ToExpr<F> for &StructuralWitIn {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::StructuralWitIn(self.id, self.max_len, self.offset, self.multi_factor)
    }
}

impl<F: PrimeField> ToExpr<F> for Fixed {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::Fixed(*self)
    }
}

impl<F: PrimeField> ToExpr<F> for &Fixed {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::Fixed(**self)
    }
}

impl<F: PrimeField> ToExpr<F> for Instance {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::InstanceScalar(*self)
    }
}

impl Instance {
    pub fn expr_as_instance<F: PrimeField>(&self) -> Expression<F> {
        Expression::Instance(*self)
    }
}

impl<F: PrimeField> ToExpr<F> for F {
    type Output = Expression<F>;
    fn expr(&self) -> Expression<F> {
        Expression::Constant(*self)
    }
}

impl<F: PrimeField> ToExpr<F> for Expression<F> {
    type Output = Expression<F>;
    fn expr(&self) -> Self::Output {
        self.clone()
    }
}

#[inline(always)]
fn eval_expr_at_index<F: PrimeField>(
    expr: &Expression<F>,
    i: usize,
    witness: &[ArcMultilinearExtension<F>],
    challenges: &[F],
) -> F {
    match expr {
        Expression::Challenge(c_id, pow, scalar, offset) => {
            challenges[*c_id as usize].pow(vec![*pow as u64]) * *scalar + *offset
        }
        Expression::Constant(c) => *c,
        Expression::WitIn(id) => *witness[*id as usize].as_ref().index(i),
        e => panic!("Unsupported expr in flat eval {:?}", e),
    }
}

/// infer full witness from flat expression over monomial terms
///
/// evaluates each term as scalar * product at every point,
/// where scalar is constant and product varies by index.
/// returns a multilinear extension of the combined result.
///
/// `witness` is assumed to be wit ++ structural_wit ++ fixed.
pub fn wit_infer_by_monomial_expr<'a, F: PrimeField>(
    flat_expr: &[Term<Expression<F>, Expression<F>>],
    witness: &[ArcMultilinearExtension<'a, F>],
    instance: &[ArcMultilinearExtension<'a, F>],
    challenges: &[F],
) -> ArcMultilinearExtension<'a, F> {
    let eval_leng = witness[0].len();

    let witness = chain!(witness, instance).cloned().collect_vec();

    // evaluate all scalar terms first
    // when instance was access in scalar, we only take its first item
    // this operation is sound
    let instance_first_element = instance
        .iter()
        .map(|instance| *instance.index(0))
        .collect_vec();
    let scalar_evals = flat_expr
        .par_iter()
        .map(|Term { scalar, .. }| {
            eval_by_expr_constant(&instance_first_element, challenges, scalar)
        })
        .collect::<Vec<_>>();

    let evaluations: Vec<_> = (0..eval_leng)
        .into_par_iter()
        .map(|i| {
            flat_expr
                .iter()
                .enumerate()
                .fold(F::ZERO, |acc, (term_index, Term { product, .. })| {
                    let scalar_val = scalar_evals[term_index];
                    let prod_val = product.iter().fold(F::ONE, |acc, e| {
                        let v = eval_expr_at_index(e, i, &witness, challenges);
                        v * acc
                    });

                    // term := scalar_val * prod_val
                    let term = scalar_val * prod_val;

                    acc + term
                })
        })
        .collect();

    DensePolynomial::new(evaluations).into()
}

/// infer witness value from expression by flattening into monomial terms
///
/// combines witnesses, structural witnesses, and fixed columns,
/// then delegates to monomial-based inference.
#[allow(clippy::too_many_arguments)]
pub fn wit_infer_by_expr<'a, F: PrimeField>(
    expr: &Expression<F>,
    n_witin: WitnessId,
    n_structural_witin: WitnessId,
    n_fixed: WitnessId,
    fixed: &[ArcMultilinearExtension<'a, F>],
    witnesses: &[ArcMultilinearExtension<'a, F>],
    structual_witnesses: &[ArcMultilinearExtension<'a, F>],
    instance: &[ArcMultilinearExtension<'a, F>],
    challenges: &[F],
) -> ArcMultilinearExtension<'a, F> {
    let witin = chain!(witnesses, structual_witnesses, fixed)
        .cloned()
        .collect_vec();
    wit_infer_by_monomial_expr(
        &monomialize_expr_to_wit_terms(expr, n_witin, n_structural_witin, n_fixed),
        &witin,
        instance,
        challenges,
    )
}

pub fn rlc_chip_record<F: PrimeField>(
    records: Vec<Expression<F>>,
    chip_record_alpha: Expression<F>,
    chip_record_beta: Expression<F>,
) -> Expression<F> {
    assert!(!records.is_empty());
    let beta_pows = power_sequence(chip_record_beta);

    let item_rlc = izip!(records, beta_pows)
        .map(|(record, beta)| record * beta)
        .sum::<Expression<F>>();

    item_rlc + chip_record_alpha.clone()
}

/// derive power sequence [1, base, base^2, ..., base^(len-1)] of base expression
pub fn power_sequence<F: PrimeField>(base: Expression<F>) -> impl Iterator<Item = Expression<F>> {
    assert!(
        matches!(
            base,
            Expression::Constant { .. } | Expression::Challenge { .. }
        ),
        "expression must be constant or challenge"
    );
    successors(Some(F::ONE.expr()), move |prev| {
        Some(prev.clone() * base.clone())
    })
}

macro_rules! impl_from_via_ToExpr {
    ($($t:ty),*) => {
        $(
            impl<F: PrimeField> From<$t> for Expression<F> {
                fn from(value: $t) -> Self {
                    value.expr()
                }
            }
        )*
    };
}
impl_from_via_ToExpr!(WitIn, Fixed, StructuralWitIn, Instance);
impl_from_via_ToExpr!(&WitIn, &Fixed, &StructuralWitIn, &Instance);

// Implement From trait for unsigned types of at most 64 bits
#[macro_export]
macro_rules! impl_expr_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl<F: PrimeField> From<$t> for Expression<F> {
                fn from(value: $t) -> Self {
                    Expression::Constant(F::from(value as u64))
                }
            }
        )*
    }
}
impl_expr_from_unsigned!(u8, u16, u32, u64, usize);

// Implement From trait for signed types
macro_rules! impl_from_signed {
    ($($t:ty),*) => {
        $(
            impl<F: PrimeField> From<$t> for Expression<F> {
                fn from(value: $t) -> Self {
                    Expression::Constant(F::from(value as i128))
                }
            }
        )*
    };
}
impl_from_signed!(i8, i16, i32, i64, isize);

impl<F: PrimeField> Display for Expression<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wtns = vec![];
        write!(f, "{}", fmt::expr(self, &mut wtns, false))
    }
}

pub mod fmt {
    use super::*;
    use std::fmt::Write;

    pub fn expr<F: PrimeField>(
        expression: &Expression<F>,
        wtns: &mut Vec<WitnessId>,
        add_parens_sum: bool,
    ) -> String {
        match expression {
            Expression::WitIn(wit_in) => {
                if !wtns.contains(wit_in) {
                    wtns.push(*wit_in);
                }
                format!("WitIn({})", wit_in)
            }
            Expression::StructuralWitIn(wit_in, max_len, offset, multi_factor) => {
                format!(
                    "StructuralWitIn({}, {}, {}, {})",
                    wit_in, max_len, offset, multi_factor
                )
            }
            Expression::Challenge(id, pow, scaler, offset) => {
                if *pow == 1 && *scaler == F::ONE && *offset == F::ZERO {
                    format!("Challenge({})", id)
                } else {
                    let mut s = String::new();
                    if *scaler != F::ONE {
                        write!(s, "{}*", scaler).unwrap();
                    }
                    write!(s, "Challenge({})", id,).unwrap();
                    if *pow > 1 {
                        write!(s, "^{}", pow).unwrap();
                    }
                    if *offset != F::ZERO {
                        write!(s, "+{}", offset).unwrap();
                    }
                    s
                }
            }
            Expression::Constant(constant) => format!("Constant({constant})"),
            Expression::Fixed(fixed) => format!("F{:?}", fixed),
            Expression::Instance(i) => format!("I{:?}", i),
            Expression::InstanceScalar(i) => format!("Is{:?}", i),
            Expression::Sum(left, right) => {
                let s = format!("{} + {}", expr(left, wtns, false), expr(right, wtns, false));
                if add_parens_sum {
                    format!("({})", s)
                } else {
                    s
                }
            }
            Expression::Product(left, right) => {
                format!("{} * {}", expr(left, wtns, true), expr(right, wtns, true))
            }
            Expression::ScaledSum(x, a, b) => {
                let s = format!(
                    "{} * {} + {}",
                    expr(a, wtns, true),
                    expr(x, wtns, true),
                    expr(b, wtns, false)
                );
                if add_parens_sum {
                    format!("({})", s)
                } else {
                    s
                }
            }
        }
    }

    pub fn parens(s: String, add_parens: bool) -> String {
        if add_parens { format!("({})", s) } else { s }
    }

    pub fn wtns<F: PrimeField>(
        wtns: &[WitnessId],
        wits_in: &[ArcMultilinearExtension<F>],
        inst_id: usize,
        wits_in_name: &[String],
    ) -> String {
        use itertools::Itertools;

        wtns.iter()
            .sorted()
            .map(|wt_id| {
                let wit = &wits_in[*wt_id as usize];
                let name = &wits_in_name[*wt_id as usize];
                let value_fmt = wit.evals_ref()[inst_id];
                format!("  WitIn({wt_id})={value_fmt} {name:?}")
            })
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        poly::dense::DensePolynomial,
        sumcheck::{WitIn, wit_infer_by_expr},
    };

    use super::{Expression, ToExpr, fmt};
    use ark_ff::{AdditiveGroup, Field};

    #[test]
    fn test_expression_arithmetics() {
        type F = ark_bn254::Fq;
        let x = WitIn { id: 0 };

        // scaledsum * challenge
        // 3 * x + 2
        let expr: Expression<F> = 3 * x.expr() + 2;
        // c^3 + 1
        let c = Expression::Challenge(0, 3, F::from(1), F::from(1));
        // res
        // x* (c^3*3 + 3) + 2c^3 + 2
        assert_eq!(
            c * expr,
            Expression::ScaledSum(
                Box::new(x.expr()),
                Box::new(Expression::Challenge(0, 3, F::from(3), F::from(3))),
                Box::new(Expression::Challenge(0, 3, F::from(2), F::from(2)))
            )
        );

        // constant * witin
        // 3 * x
        let expr: Expression<F> = 3 * x.expr();
        assert_eq!(
            expr,
            Expression::ScaledSum(
                Box::new(x.expr()),
                Box::new(Expression::Constant(F::from(3))),
                Box::new(Expression::Constant(F::from(0)))
            )
        );

        // constant * challenge
        // 3 * (c^3 + 1)
        let expr: Expression<F> = Expression::Constant(F::from(3));
        let c = Expression::Challenge(0, 3, F::from(1), F::from(1));
        assert_eq!(
            expr * c,
            Expression::Challenge(0, 3, F::from(3), F::from(3))
        );

        // challenge * challenge
        // (2c^3 + 1) * (2c^2 + 1) = 4c^5 + 2c^3 + 2c^2 + 1
        let res: Expression<F> = Expression::Challenge(0, 3, F::from(2), F::from(1))
            * Expression::Challenge(0, 2, F::from(2), F::from(1));
        assert_eq!(
            res,
            Expression::Sum(
                Box::new(Expression::Sum(
                    // (s1 * s2 * c1^(pow1 + pow2) + offset1 * offset2
                    Box::new(Expression::Challenge(
                        0,
                        3 + 2,
                        F::from(2 * 2),
                        F::ONE * F::ONE,
                    )),
                    // offset2 * s1 * c1^(pow1)
                    Box::new(Expression::Challenge(0, 3, F::from(2), F::ZERO)),
                )),
                // offset1 * s2 * c2^(pow2))
                Box::new(Expression::Challenge(0, 2, F::from(2), F::ZERO)),
            )
        );
    }

    #[test]
    fn test_is_monomial_form() {
        type F = ark_bn254::Fq;
        let x = WitIn { id: 0 };
        let y = WitIn { id: 1 };
        let z = WitIn { id: 2 };
        // scaledsum * challenge
        // 3 * x + 2
        let expr: Expression<F> = 3 * x.expr() + 2;
        assert!(expr.is_monomial_form());

        // 2 product term
        let expr: Expression<F> = 3 * x.expr() * y.expr() + 2 * x.expr();
        assert!(expr.is_monomial_form());

        // complex linear operation
        // (2c + 3) * x * y - 6z
        let expr: Expression<F> =
            Expression::Challenge(0, 1, F::from(2_u64), F::from(3_u64)) * x.expr() * y.expr()
                - 6 * z.expr();
        assert!(expr.is_monomial_form());

        // complex linear operation
        // (2c + 3) * x * y - 6z
        let expr: Expression<F> =
            Expression::Challenge(0, 1, F::from(2), F::from(3)) * x.expr() * y.expr()
                - 6 * z.expr();
        assert!(expr.is_monomial_form());

        // complex linear operation
        // (2 * x + 3) * 3 + 6 * 8
        let expr: Expression<F> = (2 * x.expr() + 3) * 3 + 6 * 8;
        assert!(expr.is_monomial_form());
    }

    #[test]
    fn test_not_monomial_form() {
        type F = ark_bn254::Fq;
        let x = WitIn { id: 0 };
        let y = WitIn { id: 1 };
        // scaledsum * challenge
        // (x + 1) * (y + 1)
        let expr: Expression<F> = (1 + x.expr()) * (2 + y.expr());
        assert!(!expr.is_monomial_form());
    }

    #[test]
    fn test_fmt_expr_challenge_1() {
        type F = ark_bn254::Fq;
        let a = Expression::<F>::Challenge(0, 2, F::from(3), F::from(4));
        let b = Expression::<F>::Challenge(0, 5, F::from(6), F::from(7));

        let mut wtns_acc = vec![];
        let s = fmt::expr(&(a * b), &mut wtns_acc, false);

        assert_eq!(
            s,
            "18*Challenge(0)^7+28 + 21*Challenge(0)^2 + 24*Challenge(0)^5"
        );
    }

    #[test]
    fn test_fmt_expr_challenge_2() {
        type F = ark_bn254::Fq;
        let a = Expression::<F>::Challenge(0, 1, F::from(1), F::from(0));
        let b = Expression::<F>::Challenge(0, 1, F::from(1), F::from(0));

        let mut wtns_acc = vec![];
        let s = fmt::expr(&(a * b), &mut wtns_acc, false);

        assert_eq!(s, "Challenge(0)^2");
    }

    #[test]
    fn test_fmt_expr_wtns_acc_1() {
        type F = ark_bn254::Fq;
        let expr = Expression::<F>::WitIn(0);
        let mut wtns_acc = vec![];
        let s = fmt::expr(&expr, &mut wtns_acc, false);
        assert_eq!(s, "WitIn(0)");
        assert_eq!(wtns_acc, vec![0]);
    }

    #[test]
    fn test_raw_wit_infer_by_monomial_expr() {
        type F = ark_bn254::Fq;
        let a = WitIn { id: 0 };
        let b = WitIn { id: 1 };
        let c = WitIn { id: 2 };

        let expr: Expression<F> = a.expr()
            + b.expr()
            + a.expr() * b.expr()
            + (c.expr() * 3 + 2)
            + Expression::Challenge(0, 1, F::ONE, F::ONE);

        let res = wit_infer_by_expr(
            &expr,
            3,
            0,
            0,
            &[],
            &[
                DensePolynomial::new(vec![F::from(1)]).into(),
                DensePolynomial::new(vec![F::from(2)]).into(),
                DensePolynomial::new(vec![F::from(3)]).into(),
            ],
            &[],
            &[],
            &[F::ONE],
        );
        assert_eq!(res.len(), 1);
    }
}
