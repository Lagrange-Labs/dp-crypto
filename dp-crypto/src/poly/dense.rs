#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
#[cfg(feature = "parallel")]
use crate::poly::eq::EQ_PARALLEL_THRESHOLD;
use anyhow::ensure;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{Rng, RngCore};
use core::ops::Index;
use either::Either;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    poly::{eq, field::mul_01_optimized, slice::SmartSlice, unsafe_allocate_zero_vec},
    util::ceil_log2,
};

/// A point is a vector of num_var length
pub type Point<F> = Vec<F>;

/// A point and the evaluation of this point.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PointAndEval<F> {
    pub point: Point<F>,
    pub eval: F,
}

impl<F> PointAndEval<F> {
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixOrder {
    LowToHigh,
    HighToLow,
}

// Field trait bound required for CanonicalSerialize and CanonicalDeserialize
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: CanonicalSerialize",
    deserialize = "F: CanonicalDeserialize"
))]
pub struct DensePolynomial<'a, F: Field> {
    num_vars: usize, // the number of variables in the multilinear polynomial
    len: usize,
    z: SmartSlice<'a, F>,
    padded_num_vars: Option<usize>,
}

macro_rules! split_eval_chunks {
    ($chunk_fn:ident, $smart_variant:ident, $smart_slice: expr, $chunk_size:expr, $num_vars:expr, $num_chunks:expr) => {{
        $smart_slice
            .$chunk_fn($chunk_size)
            .map(|chunk| {
                let chunk_len = chunk.len();
                if chunk_len == $chunk_size {
                    DensePolynomial {
                        z: SmartSlice::$smart_variant(chunk),
                        len: chunk_len,
                        num_vars: $num_vars,
                        padded_num_vars: None,
                    }
                } else {
                    let num_actual_vars = ceil_log2(chunk.len());
                    DensePolynomial {
                        z: SmartSlice::$smart_variant(chunk),
                        len: chunk_len,
                        num_vars: num_actual_vars,
                        padded_num_vars: Some($num_vars),
                    }
                }
            })
            .chain(std::iter::repeat(DensePolynomial {
                z: SmartSlice::Owned(vec![F::ZERO]),
                len: 1,
                num_vars: 0,
                padded_num_vars: Some($num_vars),
            }))
            .take($num_chunks)
            .collect::<Vec<_>>()
    }};
}

impl<'a, F: Field> DensePolynomial<'a, F> {
    pub fn new(z: Vec<F>) -> Self {
        Self::new_from_smart_slice(SmartSlice::Owned(z))
    }
    pub fn new_from_smart_slice(z: SmartSlice<'a, F>) -> Self {
        assert!(
            z.len().is_power_of_two(),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            z.len()
        );
        DensePolynomial {
            num_vars: z.len().ilog2() as usize,
            len: z.len(),
            z,
            padded_num_vars: None,
        }
    }

    pub fn shallow_clone<'b>(&'a self) -> DensePolynomial<'b, F>
    where
        'a: 'b,
    {
        Self {
            num_vars: self.num_vars,
            len: self.len,
            z: self.z.as_borrow(),
            padded_num_vars: self.padded_num_vars,
        }
    }

    /// This method allows to 0-pad a polynomial to `num_vars`, without
    /// explicitly increasing the size of the evaluations vector.
    /// The method returns an error if the provided `num_vars < self.num_vars()`
    pub fn zero_pad_num_vars(&mut self, num_vars: usize) -> anyhow::Result<()> {
        ensure!(
            num_vars >= self.num_vars,
            "Number of variables to pad the MLE ({num_vars}) must be >= than the unpadded number of variables ({})",
            self.num_vars,
        );
        // set `self.padded_num_vars` to `num_vars` only if `num_vars > self.num_vars`,
        // as otherwise we are not actually padding
        self.padded_num_vars = (num_vars > self.num_vars).then_some(num_vars);
        Ok(())
    }

    pub fn new_from_unpadded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !poly_evals.len().is_power_of_two() {
            poly_evals.push(F::zero());
        }

        Self::new_from_smart_slice(SmartSlice::Owned(poly_evals))
    }

    pub fn num_vars(&self) -> usize {
        self.padded_num_vars.unwrap_or(self.num_vars)
    }

    pub fn len(&self) -> usize {
        self.padded_num_vars.map(|nv| 1 << nv).unwrap_or(self.len)
    }

    pub fn unpadded_len(&self) -> usize {
        self.len
    }

    pub(crate) fn unpadded_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn is_padded(&self) -> bool {
        self.padded_num_vars.is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_bound(&self) -> bool {
        self.len != self.z.len()
    }

    pub fn is_mut(&self) -> bool {
        matches!(self.z, SmartSlice::BorrowedMut(_) | SmartSlice::Owned(_))
    }

    pub fn to_owned(&self) -> DensePolynomial<'static, F> {
        DensePolynomial {
            z: SmartSlice::Owned(self.z.to_vec()),
            num_vars: self.num_vars,
            len: self.len,
            padded_num_vars: self.padded_num_vars,
        }
    }

    /// Returns `Right(&mut self)` if mutable access is possible, otherwise `Left(&self)`
    pub fn to_either(&mut self) -> Either<&Self, &mut Self> {
        if self.is_mut() {
            Either::Right(self)
        } else {
            Either::Left(self)
        }
    }

    /// Relabel the point in place by switching `k` scalars from position `a` to
    /// position `b`, and from position `b` to position `a` in vector.
    ///
    /// This function turns `P(x_1,...,x_a,...,x_{a+k - 1},...,x_b,...,x_{b+k - 1},...,x_n)`
    /// to `P(x_1,...,x_b,...,x_{b+k - 1},...,x_a,...,x_{a+k - 1},...,x_n)`
    pub fn relabel_in_place(&mut self, mut a: usize, mut b: usize, k: usize) {
        assert!(
            !self.is_padded(),
            "This method is unsupported for padded polynomials"
        );
        // enforce order of a and b
        if a > b {
            ark_std::mem::swap(&mut a, &mut b);
        }
        if a == b || k == 0 {
            return;
        }
        assert!(b + k <= self.num_vars, "invalid relabel argument");
        assert!(a + k <= b, "overlapped swap window is not allowed");
        for i in 0..self.z.len() {
            let j = swap_bits(i, a, b, k);
            if i < j {
                self.z.to_mut().swap(i, j);
            }
        }
    }

    fn fix_variables_in_place<'b>(
        &mut self,
        vars: impl Iterator<Item = &'b F>,
        order: FixOrder,
        parallel: bool,
    ) {
        vars.for_each(|r| {
            if parallel {
                self.par_fix_mut(r, order)
            } else {
                self.fix_mut(r, order)
            }
        })
    }

    pub fn fix_high_variables_in_place(&mut self, vars: &[F]) {
        self.fix_variables_in_place(vars.iter().rev(), FixOrder::HighToLow, false)
    }

    pub fn fix_high_variables_in_place_parallel(&mut self, vars: &[F]) {
        self.fix_variables_in_place(vars.iter().rev(), FixOrder::HighToLow, true)
    }

    pub fn fix_low_variables_in_place(&mut self, vars: &[F]) {
        self.fix_variables_in_place(vars.iter(), FixOrder::LowToHigh, false);
    }

    pub fn fix_low_variables_in_place_parallel(&mut self, vars: &[F]) {
        self.fix_variables_in_place(vars.iter(), FixOrder::LowToHigh, true);
    }

    pub fn fix_high_variables(&self, vars: &[F]) -> Self {
        if vars.is_empty() {
            return self.clone();
        }
        // fix the first high variable
        let mut fixed_poly = self.new_fix_top(&vars[vars.len() - 1]);
        // then iterate over the remaining variables using the mutable variant
        fixed_poly.fix_high_variables_in_place(&vars[..vars.len() - 1]);
        fixed_poly
    }

    pub fn fix_high_variables_parallel(&self, vars: &[F]) -> Self {
        if vars.is_empty() {
            return self.clone();
        }
        // fix the first high variable
        let mut fixed_poly = self.new_fix_top(&vars[vars.len() - 1]);
        // then iterate over the remaining variables using the mutable variant
        fixed_poly.fix_high_variables_in_place_parallel(&vars[..vars.len() - 1]);
        fixed_poly
    }

    pub fn fix_low_variables(&self, vars: &[F]) -> Self {
        if vars.is_empty() {
            return self.clone();
        }
        // fix the first high variable
        let mut fixed_poly = self.fix_low(&vars[0]);
        // then iterate over the remaining variables using the mutable variant
        fixed_poly.fix_low_variables_in_place(&vars[1..]);
        fixed_poly
    }

    pub fn fix_low_variables_parallel(&self, vars: &[F]) -> Self {
        if vars.is_empty() {
            return self.clone();
        }
        // fix the first high variable
        let mut fixed_poly = self.fix_low_parallel(&vars[0]);
        // then iterate over the remaining variables using the mutable variant
        fixed_poly.fix_low_variables_in_place_parallel(&vars[1..]);
        fixed_poly
    }

    pub fn fix_mut(&mut self, r: &F, order: FixOrder) {
        match order {
            FixOrder::LowToHigh => self.fix_low_mut(r),
            FixOrder::HighToLow => self.fix_high_mut(r),
        }
    }

    pub fn par_fix_mut(&mut self, r: &F, order: FixOrder) {
        match order {
            FixOrder::LowToHigh => self.fix_low_mut_parallel(r),
            FixOrder::HighToLow => self.par_fix_mut_top(r),
        }
    }

    fn fix_high_mut_generic<FU: Fn(&mut Self, &F)>(&mut self, r: &F, fix_unpadded: FU) {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        if self.len() > self.unpadded_len() {
            // if the polynomial is 0-padded to an higher power of 2 length, then the second
            // half of the evaluations are 0. Therefore, to fix the polynomial, we just need
            // to multiply by `1-r` the non-zero variables
            #[cfg(not(feature = "parallel"))]
            let it = self.z.iter_mut();
            #[cfg(feature = "parallel")]
            let it = self.z.par_iter_mut().with_min_len(4096);
            it.for_each(|a| {
                *a *= F::ONE - r;
            });
        } else {
            fix_unpadded(self, r)
        }
        if let Some(nv) = self.padded_num_vars.as_mut() {
            *nv -= 1;
        }
    }

    pub fn fix_high_mut(&mut self, r: &F) {
        self.fix_high_mut_generic(r, |poly, r| {
            let n = poly.len() / 2;
            let (left, right) = poly.z.to_mut().split_at_mut(n);

            #[cfg(not(feature = "parallel"))]
            let it = left.iter_mut().zip(right.iter());
            #[cfg(feature = "parallel")]
            let it = left.par_iter_mut().zip(right.par_iter()).with_min_len(4096);
            it.for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });
            poly.z.truncate_mut(n);
            poly.num_vars -= 1;
            poly.len = n;
        })
    }

    pub fn fix_high_many_ones_top(&mut self, r: &F) {
        self.fix_high_mut_generic(r, |poly, r| {
            let n = poly.len() / 2;
            let (left, right) = poly.z.to_mut().split_at_mut(n);

            #[cfg(not(feature = "parallel"))]
            let it = left.iter_mut().zip(right.iter());
            #[cfg(feature = "parallel")]
            let it = left.par_iter_mut().zip(right.par_iter()).with_min_len(4096);

            it.filter(|&(&mut a, &b)| a != b).for_each(|(a, b)| {
                let m = *b - *a;
                if m.is_one() {
                    *a += *r;
                } else {
                    *a += *r * m;
                }
            });
            poly.z.truncate_mut(n);
            poly.num_vars -= 1;
            poly.len = n;
        })
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn par_fix_mut_top(&mut self, r: &F) {
        self.fix_high_mut_generic(r, |poly, r| {
            let n = poly.len() / 2;

            let (left, right) = poly.z.to_mut().split_at_mut(n);

            #[cfg(not(feature = "parallel"))]
            let it = left.iter_mut().zip(right.iter());
            #[cfg(feature = "parallel")]
            let it = left.par_iter_mut().zip(right.par_iter()).with_min_len(4096);

            it.filter(|&(&mut a, &b)| a != b).for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });
            poly.z.truncate_mut(n);
            poly.num_vars -= 1;
            poly.len = n;
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn new_fix_top(&self, r: &F) -> Self {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        if self.len() > self.unpadded_len() {
            // if the polynomial is 0-padded to an higher power of 2 length, then the second
            // half of the evaluations are 0. Therefore, to fix the polynomial, we just need
            // to multiply by `1-r` the non-zero variables
            let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(self.unpadded_len());

            #[cfg(not(feature = "parallel"))]
            let it = new_evals.iter_mut().zip(self.z.iter());
            #[cfg(feature = "parallel")]
            let it = new_evals
                .par_iter_mut()
                .zip(self.z.par_iter())
                .with_min_len(4096);
            it.for_each(|(new, old)| {
                *new = *old * (F::ONE - r);
            });
            Self {
                num_vars: self.num_vars,
                len: self.len,
                z: SmartSlice::Owned(new_evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        } else {
            let n = self.len() / 2;
            let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

            #[cfg(not(feature = "parallel"))]
            let it = new_evals.iter_mut().zip(0..n);
            #[cfg(feature = "parallel")]
            let it = new_evals.par_iter_mut().zip(0..n).with_min_len(4096);

            it.for_each(|(new, i)| {
                // let low' = low + r * (high - low)
                let low = self.z[i];
                let high = self.z[i + n];
                let m = high - low;
                *new = low + *r * m;
            });
            let num_vars = self.num_vars - 1;
            let len = n;

            Self {
                num_vars,
                len,
                z: SmartSlice::Owned(new_evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        }
    }

    /// Note: does not truncate
    #[tracing::instrument(skip_all)]
    pub fn fix_low_mut(&mut self, r: &F) {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        let len = self.unpadded_len();
        if len == 1 {
            self.z[0] *= F::ONE - r
        } else {
            let n = len / 2;
            for i in 0..n {
                self.z[i] = self.z[2 * i] + *r * (self.z[2 * i + 1] - self.z[2 * i]);
            }
            self.z.truncate_mut(n);
            self.num_vars -= 1;
            self.len = n;
        }

        if let Some(nv) = self.padded_num_vars.as_mut() {
            *nv -= 1
        }
    }

    pub fn fix_low(&self, r: &F) -> Self {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        let len = self.unpadded_len();
        if len == 1 {
            let new_evals = vec![self.z[0] * (F::ONE - r)];
            Self {
                num_vars: self.num_vars,
                len,
                z: SmartSlice::Owned(new_evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        } else {
            let n = len / 2;
            let mut new_evals = unsafe_allocate_zero_vec(n);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let low = self.z[2 * i];
                let high = self.z[2 * i + 1];
                let m = high - low;
                new_evals[i] = low + *r * m;
            }
            Self {
                num_vars: self.num_vars - 1,
                len: n,
                z: SmartSlice::Owned(new_evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        }
    }

    pub fn bound_poly_var_bot_01_inplace(&mut self, r: &F) {
        let n = self.unpadded_len() / 2;

        for i in 0..n {
            let a0 = self.z[2 * i];
            let a1 = self.z[2 * i + 1];
            let m = a1 - a0;

            self.z[i] = if m.is_zero() {
                a0
            } else if m.is_one() {
                a0 + *r
            } else {
                a0 + *r * m
            };
        }

        self.z.truncate_mut(n);
    }

    pub fn bound_poly_var_bot_01_gap(&mut self, r: &F) {
        let n = self.unpadded_len() / 2;

        self.z.par_chunks_exact_mut(2)
            .with_min_len(512)
            .for_each(|chunk| {
                let a0 = chunk[0];
                let a1 = chunk[1];
                chunk[0] = a0 + *r * (a1 - a0);
            });

        // Phase 2: sequential compaction
        for i in 0..n {
            self.z[i] = self.z[2 * i];
        }

        self.z.truncate_mut(n);
    }

    pub fn bound_poly_var_bot_01_optimized(&self, r: &F) -> Vec<F> {
        let n = self.unpadded_len() / 2;
        let mut bound_z: Vec<F> = unsafe_allocate_zero_vec(n);
        unsafe { bound_z.set_len(0) };
        (bound_z.spare_capacity_mut(), self.z.par_chunks_exact(2))
            .into_par_iter()
            .with_min_len(512)
            .for_each(|(bound_coeff, coeffs)| {
                let m = coeffs[1] - coeffs[0];
                bound_coeff.write(if m.is_zero() {
                    coeffs[0]
                } else if m.is_one() {
                    coeffs[0] + *r
                } else {
                    coeffs[0] + *r * m
                });
            });
        unsafe { bound_z.set_len(n) };
        bound_z
    }

    pub fn fix_low_parallel(&self, r: &F) -> Self {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        let len = self.unpadded_len();
        if len == 1 {
            let new_evals = vec![self.z[0] * (F::ONE - r)];
            Self {
                num_vars: self.num_vars,
                len,
                z: SmartSlice::Owned(new_evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        } else {
            let evals = self.bound_poly_var_bot_01_optimized(r);
            Self {
                num_vars: self.num_vars - 1,
                len: evals.len(),
                z: SmartSlice::Owned(evals),
                padded_num_vars: self.padded_num_vars.map(|nv| nv - 1),
            }
        }
    }

    pub fn fix_low_mut_parallel(&mut self, r: &F) {
        assert!(
            self.num_vars() > 0,
            "Cannot fix variable on a constant polynomial"
        );
        let len = self.unpadded_len();
        if len == 1 {
            self.z[0] *= F::ONE - r;
        } else {
            let evals = self.bound_poly_var_bot_01_optimized(r);
            self.num_vars -= 1;
            self.len = evals.len();
            self.z = SmartSlice::Owned(evals);
        }
        if let Some(nv) = self.padded_num_vars.as_mut() {
            *nv -= 1
        }
    }

    pub fn evaluate_dot_product(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.num_vars);
        let chis = eq::evals(r);
        assert_eq!(chis.len(), self.z.len());
        self.evaluate_at_chi(&chis)
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> anyhow::Result<F> {
        ensure!(
            r.len() == self.num_vars(),
            "r len() = {} vs num_vars = {}",
            r.len(),
            self.num_vars()
        );
        // portion of the point dealing with unpadded number of variables
        let (r_unpad, r_padded) = r.split_at(self.unpadded_num_vars());
        //let r = r_vec.as_slice();
        let m = r_unpad.len() / 2;
        let (r2, r1) = r_unpad.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| eq::evals(r1), || eq::evals(r2));
        let eval = self.split_eq_evaluate(&eq_one, &eq_two);
        // add padding coordinates to `eval`
        Ok(r_padded.iter().fold(eval, |acc, r| acc * (F::ONE - r)))
    }

    pub fn split_eq_evaluate(&self, eq_one: &[F], eq_two: &[F]) -> F {
        #[cfg(feature = "parallel")]
        {
            let r_len = eq_one.len() + eq_two.len();
            if r_len < EQ_PARALLEL_THRESHOLD {
                self.evaluate_split_eq_serial(eq_one, eq_two)
            } else {
                self.evaluate_split_eq_parallel(eq_one, eq_two)
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.evaluate_split_eq_serial(eq_one, eq_two)
        }
    }
    fn evaluate_split_eq_parallel(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .into_par_iter()
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        mul_01_optimized(eq_two[x2], self.z[idx])
                    })
                    .reduce(|| F::zero(), |acc, val| acc + val);
                mul_01_optimized(eq_one[x1], partial_sum)
            })
            .reduce(|| F::zero(), |acc, val| acc + val);
        eval
    }

    fn evaluate_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        mul_01_optimized(eq_two[x2], self.z[idx])
                    })
                    .fold(F::zero(), |acc, val| acc + val);
                mul_01_optimized(eq_one[x1], partial_sum)
            })
            .fold(F::zero(), |acc, val| acc + val);
        eval
    }

    // Faster evaluation based on
    // https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
    // Shaves a factor of 2 from run time.
    pub fn inside_out_evaluate(&self, r: &[F]) -> F {
        // portion of the point dealing with unpadded number of variables
        let (r_unpad, r_padded) = r.split_at(self.unpadded_num_vars());
        // Copied over from eq_poly
        // If the number of variables are greater
        // than 2^16 -- use parallel evaluate
        // Below that it's better to just do things linearly.
        const PARALLEL_THRESHOLD: usize = 16;

        // r must have a value for each variable
        assert_eq!(r_unpad.len(), self.num_vars);
        let eval = {
            #[cfg(feature = "parallel")]
            {
                let m = r_unpad.len();
                if m < PARALLEL_THRESHOLD {
                    self.inside_out_serial(r_unpad)
                } else {
                    self.inside_out_parallel(r_unpad)
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.inside_out_serial(r_unpad)
            }
        };
        // add padding coordinates to `eval`
        r_padded.iter().fold(eval, |acc, r| acc * (F::ONE - r))
    }

    fn inside_out_serial(&self, r: &[F]) -> F {
        // r is expected to be big endinan
        // r[0] is the most significant digit
        let mut current = self.z.to_vec();
        let m = r.len();
        for i in 0..m {
            let stride = 1 << (m - 1 - i);

            // Note that as r is big endian
            // and i is reversed
            // r[m-1-i] actually starts at the big endian digit
            // and moves towards the little endian digit.
            for j in 0..stride {
                let f0 = current[j];
                let f1 = current[j + stride];
                let slope = f1 - f0;
                if slope.is_zero() {
                    current[j] = f0;
                } else if slope.is_one() {
                    current[j] = f0 + r[m - 1 - i];
                } else {
                    current[j] = f0 + r[m - 1 - i] * slope;
                }
            }
            // No benefit to truncating really.
            //current.truncate(stride);
        }
        current[0]
    }

    fn inside_out_parallel(&self, r: &[F]) -> F {
        let mut current: Vec<_> = self.z.par_iter().cloned().collect();
        let m = r.len();
        // Invoking the same parallelisation structure
        // currently in evaluating in Lagrange bases.
        // See eq_poly::evals()
        for i in 0..m {
            let stride = 1 << (m - 1 - i);
            let r_val = r[m - 1 - i];
            let (evals_left, evals_right) = current.split_at_mut(stride);
            let (evals_right, _) = evals_right.split_at_mut(stride);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter())
                .for_each(|(x, y)| {
                    let slope = *y - *x;
                    if slope.is_zero() {
                        return;
                    }
                    if slope.is_one() {
                        *x += r_val;
                    } else {
                        *x += r_val * slope;
                    }
                });
        }
        current[0]
    }
    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        compute_dotproduct(&self.z, chis)
    }

    pub fn evaluate_at_chi_low_optimized(&self, chis: &[F]) -> F {
        assert_eq!(self.z.len(), chis.len());
        compute_dotproduct_low_optimized(&self.z, chis)
    }

    pub fn evals(&self) -> Vec<F> {
        let padded_len = self.len();
        let mut evals = self.z[..self.len].to_vec();
        evals.append(&mut vec![F::ZERO; padded_len - self.len]);
        evals
    }

    pub fn evals_ref(&self) -> &[F] {
        assert!(
            !self.is_padded(),
            "Cannot return evals as slice if the MLE is padded, use evals instead"
        );
        &self.z.as_slice()[..self.len]
    }

    pub fn eval_as_univariate(&self, r: &F) -> F {
        let mut output = self.z[0];
        let mut rpow = *r;
        for z in self.z.iter().take(self.len).skip(1) {
            output += rpow * z;
            rpow *= r;
        }
        output
    }

    pub fn random<R: Rng + RngCore>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::new(
            std::iter::from_fn(|| Some(F::rand(&mut rng)))
                .take(1 << num_vars)
                .collect(),
        )
    }

    pub fn as_view(&self) -> DensePolynomial<'_, F> {
        self.as_view_slice(1, 0)
    }

    /// get mle with arbitrary start end
    pub fn as_view_slice(&self, num_chunks: usize, chunk_index: usize) -> DensePolynomial<'_, F> {
        let total_len = self.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0
                && total_len.is_multiple_of(num_chunks)
                && chunk_size > 0
                && chunk_index < num_chunks,
            "invalid num_chunks: {num_chunks} total_len: {total_len}, chunk_index {chunk_index} parameter set"
        );
        let start = chunk_size * chunk_index;

        if start < self.unpadded_len() {
            let end = self.unpadded_len().min(start + chunk_size);
            let sub_evaluations = SmartSlice::Borrowed(&self.z[start..end]);
            let num_vars = ceil_log2(end - start);
            if sub_evaluations.len() < chunk_size {
                // there is part of the 0-padding in the chunk
                DensePolynomial {
                    num_vars,
                    len: sub_evaluations.len(),
                    z: sub_evaluations,
                    padded_num_vars: Some(ceil_log2(chunk_size)),
                }
            } else {
                DensePolynomial {
                    num_vars,
                    len: chunk_size,
                    z: sub_evaluations,
                    padded_num_vars: None,
                }
            }
        } else {
            // the chunk is entirely in 0-padded area
            DensePolynomial {
                num_vars: 0,
                len: 1,
                z: SmartSlice::Owned(vec![F::ZERO]),
                padded_num_vars: Some(ceil_log2(chunk_size)),
            }
        }
    }

    pub fn as_view_mut(&mut self) -> DensePolynomial<'_, F> {
        self.as_view_slice_mut(1, 0)
    }

    /// get mle with arbitrary start end
    pub fn as_view_slice_mut(
        &mut self,
        num_chunks: usize,
        chunk_index: usize,
    ) -> DensePolynomial<'_, F> {
        let total_len = self.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0
                && total_len.is_multiple_of(num_chunks)
                && chunk_size > 0
                && chunk_index < num_chunks,
            "invalid num_chunks: {num_chunks} total_len: {total_len}, chunk_index {chunk_index} parameter set"
        );
        let start = chunk_size * chunk_index;

        if start < self.unpadded_len() {
            let end = self.unpadded_len().min(start + chunk_size);
            let sub_evaluations = SmartSlice::BorrowedMut(&mut self.z[start..end]);
            let num_vars = ceil_log2(end - start);
            if sub_evaluations.len() < chunk_size {
                // there is part of the 0-padding in the chunk
                DensePolynomial {
                    num_vars,
                    len: sub_evaluations.len(),
                    z: sub_evaluations,
                    padded_num_vars: Some(ceil_log2(chunk_size)),
                }
            } else {
                DensePolynomial {
                    num_vars,
                    len: chunk_size,
                    z: sub_evaluations,
                    padded_num_vars: None,
                }
            }
        } else {
            // the chunk is entirely in 0-padded area
            DensePolynomial {
                num_vars: 0,
                len: 1,
                z: SmartSlice::Owned(vec![F::ZERO]),
                padded_num_vars: Some(ceil_log2(chunk_size)),
            }
        }
    }

    /// splits the MLE into `num_chunks` parts, where each part contains disjoint mutable pointers
    /// to the original data (either borrowed mutably or owned).
    pub fn as_view_chunks_mut(&'a mut self, num_chunks: usize) -> Vec<DensePolynomial<'a, F>> {
        let total_len = self.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0 && total_len.is_multiple_of(num_chunks) && chunk_size > 0,
            "invalid num_chunks: {num_chunks} total_len: {total_len} parameter set"
        );
        // safety check that `chunk_size` is a power of 2;
        // it should always hold since `total_len` is power of 2 and `num_chunks` is a divisor of `total_len`
        debug_assert!(chunk_size.is_power_of_two());
        let num_vars_per_chunk = self.num_vars() - ceil_log2(num_chunks);

        split_eval_chunks!(
            chunks_mut,
            BorrowedMut,
            &mut self.z,
            chunk_size,
            num_vars_per_chunk,
            num_chunks
        )
    }

    /// immutable counterpart to [`as_view_chunks_mut`]
    pub fn as_view_chunks<'b>(&'a self, num_chunks: usize) -> Vec<DensePolynomial<'b, F>>
    where
        'a: 'b,
    {
        let total_len = self.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0 && total_len.is_multiple_of(num_chunks) && chunk_size > 0,
            "invalid num_chunks: {num_chunks} total_len: {total_len} parameter set"
        );
        // safety check that `chunk_size` is a power of 2;
        // it should always hold since `total_len` is power of 2 and `num_chunks` is a divisor of `total_len`
        debug_assert!(chunk_size.is_power_of_two());
        let num_vars_per_chunk = self.num_vars() - ceil_log2(num_chunks);

        split_eval_chunks!(
            chunks,
            Borrowed,
            self.z,
            chunk_size,
            num_vars_per_chunk,
            num_chunks
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn linear_combination(polynomials: &[&Self], coefficients: &[F]) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let max_length = polynomials.iter().map(|poly| poly.len()).max().unwrap();

        let result: Vec<F> = (0..max_length)
            .into_par_iter()
            .map(|i| {
                let mut acc = F::zero();
                for (coeff, poly) in coefficients.iter().zip(polynomials.iter()) {
                    if i < poly.unpadded_len() {
                        acc += poly.z[i] * *coeff;
                    }
                }
                acc
            })
            .collect();
        DensePolynomial::new(result)
    }
}

impl<'a, F: Field> Index<usize> for DensePolynomial<'a, F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, index: usize) -> &F {
        &(self.z[index])
    }
}

impl<'a, F: Field, N> From<&[N]> for DensePolynomial<'a, F>
where
    F: From<N>,
    N: Copy,
{
    fn from(values: &[N]) -> Self {
        DensePolynomial::new(values.iter().map(|v| F::from(*v)).collect())
    }
}

impl<'a, F: Field, N> From<Vec<N>> for DensePolynomial<'a, F>
where
    F: From<N>,
{
    fn from(values: Vec<N>) -> Self {
        DensePolynomial::new(values.into_iter().map(|v| F::from(v)).collect())
    }
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: Field>(a: &[F], b: &[F]) -> F {
    #[cfg(not(feature = "parallel"))]
    let it = a.iter().zip(b);
    #[cfg(feature = "parallel")]
    let it = a.par_iter().zip(b.par_iter()).with_min_len(4096);

    it.map(|(&a_i, &b_i)| a_i * b_i).sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    #[cfg(not(feature = "parallel"))]
    let it = a.iter().zip(b);
    #[cfg(feature = "parallel")]
    let it = a.par_iter().zip(b).with_min_len(4096);

    it.map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i)).sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: Field>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

/// swap the bits of `x` from position `a..a+n` to `b..b+n` and from `b..b+n` to `a..a+n` in little endian order
pub(crate) const fn swap_bits(x: usize, a: usize, b: usize, n: usize) -> usize {
    let a_bits = (x >> a) & ((1usize << n) - 1);
    let b_bits = (x >> b) & ((1usize << n) - 1);
    let local_xor_mask = a_bits ^ b_bits;
    let global_xor_mask = (local_xor_mask << a) | (local_xor_mask << b);
    x ^ global_xor_mask
}

#[cfg(test)]
mod tests {
    use crate::poly::{Math, challenge};
    use ark_ff::{AdditiveGroup, PrimeField};
    use ark_std::rand::{Rng, SeedableRng, thread_rng};
    use itertools::Itertools;
    use rand_chacha::ChaCha20Rng;
    use rstest::rstest;

    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    pub fn compute_chis_at_r<F: Field>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in (0..r.len()).rev() {
                let bit_j = (i & (1 << j)) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>()
    }

    fn check_memoized_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(challenge::new_from_u128(prng.r#gen::<u128>()).unwrap());
        }
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = eq::evals(&r);
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly
                .evaluate(vec![Fr::from(3), Fr::from(4)].as_slice())
                .unwrap(),
            Fr::from(8)
        );
    }
    #[test]
    fn compare_random_evaluations() {
        // Compares optimised polynomial evaluation
        // with the old polynomial evaluation
        use ark_std::rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(42);

        for &exp in &[2, 4, 6, 8] {
            let num_evals = 1 << exp; // must be a power of 2
            let num_vars = exp;

            // Generate random coefficients for the multilinear polynomial
            let evals: Vec<Fr> = (0..num_evals).map(|_| Fr::rand(&mut rng)).collect();
            let poly = DensePolynomial::<Fr>::new(evals);

            // Try 10 random evaluation points
            for _ in 0..10 {
                let eval_point: Vec<Fr> = (0..num_vars)
                    .map(|_| challenge::random_challenge(&mut rng))
                    .collect();

                let eval1 = poly.evaluate(&eval_point).unwrap();
                let eval2 = poly.inside_out_evaluate(
                    &eval_point, /*.iter()
                                 .copied()
                                 .rev()
                                 .collect::<Vec<_>>()
                                 .as_slice()*/
                );

                assert_eq!(
                    eval1, eval2,
                    "Mismatch at point {:?} for num_vars = {}: eval = {:?}, opt = {:?}",
                    eval_point, num_vars, eval1, eval2
                );
            }
        }
    }

    #[rstest]
    #[case::parallel(true)]
    #[case::base(false)]
    fn test_fix_variables(#[case] parallel_fix: bool) {
        let mut rng = ChaCha20Rng::seed_from_u64(24);

        let num_vars = 4;

        let mut poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let random_point: Point<Fr> = (0..num_vars)
            .map(|_| challenge::random_challenge(&mut rng))
            .collect();

        let eval = poly.evaluate(&random_point).unwrap();
        let fixed_vars = num_vars / 2;
        let fixed_poly = if parallel_fix {
            poly.fix_low_variables_parallel(&random_point[0..fixed_vars])
        } else {
            poly.fix_low_variables(&random_point[0..fixed_vars])
        };

        let eval_fixed = fixed_poly.evaluate(&random_point[fixed_vars..]).unwrap();

        assert_eq!(eval, eval_fixed);

        // fix in place
        if parallel_fix {
            poly.fix_low_variables_in_place_parallel(&random_point[0..fixed_vars]);
        } else {
            poly.fix_low_variables_in_place(&random_point[0..fixed_vars]);
        }

        let eval_fixed = poly.evaluate(&random_point[fixed_vars..]).unwrap();

        assert_eq!(eval, eval_fixed);

        // test fixing high variable

        let mut poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let eval = poly.evaluate(&random_point).unwrap();
        let fixed_poly = if parallel_fix {
            poly.fix_high_variables_parallel(&random_point[fixed_vars..])
        } else {
            poly.fix_high_variables(&random_point[fixed_vars..])
        };

        let eval_fixed = fixed_poly.evaluate(&random_point[..fixed_vars]).unwrap();

        assert_eq!(eval, eval_fixed);

        // fix in place
        if parallel_fix {
            poly.fix_high_variables_in_place_parallel(&random_point[fixed_vars..]);
        } else {
            poly.fix_high_variables_in_place(&random_point[fixed_vars..]);
        }

        let eval_fixed = poly.evaluate(&random_point[..fixed_vars]).unwrap();

        assert_eq!(eval, eval_fixed);
    }

    #[test]
    fn test_eval_endianness() {
        let num_evals = 4;
        let evals = (0..num_evals).map(Fr::from).collect();
        let poly = DensePolynomial::<Fr>::new(evals);

        let eval_at_1 = poly.evaluate(&[Fr::ONE, Fr::ZERO]).unwrap();
        let eval_at_2 = poly.evaluate(&[Fr::ZERO, Fr::ONE]).unwrap();

        assert_eq!(eval_at_1, Fr::from(1));
        assert_eq!(eval_at_2, Fr::from(2));
    }

    #[test]
    fn test_poly_serialization() {
        let num_vars = 5;
        let poly = DensePolynomial::<Fr>::random(num_vars, &mut thread_rng());

        let serialized_poly = rmp_serde::to_vec(&poly).unwrap();
        let deserialized_poly: DensePolynomial<Fr> =
            rmp_serde::from_slice(&serialized_poly).unwrap();

        assert_eq!(poly, deserialized_poly);

        // check with a non-owned poly
        let mut evals: Vec<_> = (0..(1 << num_vars))
            .map(|_| Fr::rand(&mut thread_rng()))
            .collect();
        let poly = DensePolynomial::<Fr>::new_from_smart_slice(SmartSlice::Borrowed(&evals));

        let serialized_poly = rmp_serde::to_vec(&poly).unwrap();
        let deserialized_poly: DensePolynomial<Fr> =
            rmp_serde::from_slice(&serialized_poly).unwrap();

        assert_eq!(poly, deserialized_poly);

        // serialize polynomial with mutable reference
        let poly = DensePolynomial::<Fr>::new_from_smart_slice(SmartSlice::BorrowedMut(&mut evals));

        let serialized_poly = rmp_serde::to_vec(&poly).unwrap();
        let deserialized_poly: DensePolynomial<Fr> =
            rmp_serde::from_slice(&serialized_poly).unwrap();

        assert_eq!(poly, deserialized_poly);
    }

    #[test]
    fn test_padded_polys() {
        let nv = 4;
        let padded_nv = 10;
        let mut poly = DensePolynomial::random(nv, &mut thread_rng());

        let mut poly_view = poly.as_view();
        poly_view.zero_pad_num_vars(padded_nv).unwrap();
        let point = (0..padded_nv)
            .map(|_| Fr::rand(&mut thread_rng()))
            .collect_vec();

        let poly_eval = poly.evaluate(&point[..nv]).unwrap();

        let padded_poly_eval = poly_view.evaluate(&point).unwrap();

        let expected_eval = (nv..padded_nv).fold(poly_eval, |eval, i| eval * (Fr::ONE - point[i]));

        assert_eq!(padded_poly_eval, expected_eval);

        let padded_poly_eval = poly_view.inside_out_evaluate(&point);

        assert_eq!(padded_poly_eval, expected_eval);

        // test fixing
        let padded_poly_eval = point
            .iter()
            .fold(poly_view.clone(), |fixed_poly, p| {
                fixed_poly.fix_low_parallel(p)
            })
            .evals()[0];

        assert_eq!(padded_poly_eval, expected_eval);

        let padded_poly_eval = point
            .iter()
            .fold(poly_view.clone(), |fixed_poly, p| fixed_poly.fix_low(p))
            .evals()[0];

        assert_eq!(padded_poly_eval, expected_eval);

        // test fix high
        let padded_poly_eval = point
            .iter()
            .rev() // we need to start from high variables since we are fixing high
            .fold(poly_view.clone(), |fixed_poly, p| fixed_poly.new_fix_top(p))
            .evals()[0];

        assert_eq!(padded_poly_eval, expected_eval);

        let padded_poly_eval = {
            point.iter().for_each(|p| poly_view.fix_low_mut_parallel(p));
            poly_view.evals()[0]
        };

        assert_eq!(padded_poly_eval, expected_eval);

        // keep a copy to re-instantiate polynomial to test fix_high_mut
        let mut poly_copy = poly.clone();

        let mut poly_view = poly.as_view_mut();
        poly_view.zero_pad_num_vars(padded_nv).unwrap();

        let padded_poly_eval = {
            point.iter().for_each(|p| poly_view.fix_low_mut(p));
            poly_view.evals()[0]
        };

        assert_eq!(padded_poly_eval, expected_eval);

        assert_eq!(poly.evals()[0], expected_eval);

        assert_ne!(poly_copy.evals()[0], expected_eval);

        let mut poly_view = poly_copy.as_view_mut();
        poly_view.zero_pad_num_vars(padded_nv).unwrap();

        let padded_poly_eval = {
            point
                .iter()
                .rev()
                .for_each(|p| poly_view.par_fix_mut_top(p));
            poly_view.evals()[0]
        };

        assert_eq!(padded_poly_eval, expected_eval);
    }
}
