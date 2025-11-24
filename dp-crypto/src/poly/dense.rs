#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
#[cfg(feature = "parallel")]
use crate::poly::eq::EQ_PARALLEL_THRESHOLD;
use anyhow::ensure;
use ark_ff::Field;
use ark_std::rand::{Rng, RngCore};
use core::ops::Index;
use either::Either;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

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
#[derive(Clone, Default, Debug, PartialEq)]
pub struct DensePolynomial<'a, F: Field> {
    pub num_vars: usize, // the number of variables in the multilinear polynomial
    pub len: usize,
    pub z: SmartSlice<'a, F>,
}

macro_rules! split_eval_chunks {
    ($chunk_fn:ident, $smart_variant:ident, $smart_slice: expr, $chunk_size:expr, $num_vars:expr) => {{
        $smart_slice
            .$chunk_fn($chunk_size)
            .map(|chunk| DensePolynomial {
                z: SmartSlice::$smart_variant(chunk),
                len: $chunk_size,
                num_vars: $num_vars,
            })
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
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !poly_evals.len().is_power_of_two() {
            poly_evals.push(F::zero());
        }

        Self::new_from_smart_slice(SmartSlice::Owned(poly_evals))
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
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

    pub fn fix_mut(&mut self, r: &F, order: FixOrder) {
        match order {
            FixOrder::LowToHigh => self.fix_low_mut(r),
            FixOrder::HighToLow => self.fix_high_mut(r),
        }
    }

    pub fn par_fix_mut(&mut self, r: F, order: FixOrder) {
        match order {
            FixOrder::LowToHigh => self.fix_low_mut_parallel(&r),
            FixOrder::HighToLow => self.par_fix_mut_top(&r),
        }
    }

    pub fn fix_high_mut(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.z.to_mut().split_at_mut(n);

        #[cfg(not(feature = "parallel"))]
        let it = left.iter_mut().zip(right.iter());
        #[cfg(feature = "parallel")]
        let it = left.par_iter_mut().zip(right.par_iter()).with_min_len(4096);
        it.for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn fix_high_many_ones_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.z.to_mut().split_at_mut(n);

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

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn par_fix_mut_top(&mut self, r: &F) {
        let n = self.len() / 2;

        let (left, right) = self.z.to_mut().split_at_mut(n);

        #[cfg(not(feature = "parallel"))]
        let it = left.iter_mut().zip(right.iter());
        #[cfg(feature = "parallel")]
        let it = left.par_iter_mut().zip(right.par_iter()).with_min_len(4096);

        it.filter(|&(&mut a, &b)| a != b).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all)]
    pub fn new_fix_top(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            // let low' = low + r * (high - low)
            let low = self.z[i];
            let high = self.z[i + n];
            let m = high - low;
            new_evals[i] = low + *r * m;
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            z: SmartSlice::Owned(new_evals),
        }
    }

    /// Note: does not truncate
    #[tracing::instrument(skip_all)]
    pub fn fix_low_mut(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.z[i] = self.z[2 * i] + *r * (self.z[2 * i + 1] - self.z[2 * i]);
        }
        self.num_vars -= 1;
        self.len = n;
    }

    pub fn fix_low(&self, r: &F) -> Self {
        let n = self.len() / 2;
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
        }
    }

    fn bound_poly_var_bot_01_optimized(&self, r: &F) -> Vec<F> {
        let n = self.len() / 2;
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
        let evals = self.bound_poly_var_bot_01_optimized(r);
        Self {
            num_vars: self.num_vars - 1,
            len: evals.len(),
            z: SmartSlice::Owned(evals),
        }
    }

    pub fn fix_low_mut_parallel(&mut self, r: &F) {
        let evals = self.bound_poly_var_bot_01_optimized(r);
        self.num_vars -= 1;
        self.len = evals.len();
        self.z = SmartSlice::Owned(evals);
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
            r.len() == self.num_vars,
            "r len() = {} vs num_vars = {}",
            r.len(),
            self.num_vars
        );
        let m = r.len() / 2;
        let (r2, r1) = r.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| eq::evals(r2), || eq::evals(r1));
        Ok(self.split_eq_evaluate(&eq_one, &eq_two))
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
        // Copied over from eq_poly
        // If the number of variables are greater
        // than 2^16 -- use parallel evaluate
        // Below that it's better to just do things linearly.
        const PARALLEL_THRESHOLD: usize = 16;

        // r must have a value for each variable
        assert_eq!(r.len(), self.num_vars);
        #[cfg(feature = "parallel")]
        {
            let m = r.len();
            if m < PARALLEL_THRESHOLD {
                self.inside_out_serial(r)
            } else {
                self.inside_out_parallel(r)
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.inside_out_serial(r)
        }
    }

    fn inside_out_serial(&self, r: &[F]) -> F {
        // r is expected to be big endinan
        // r[0] is the most significant digit
        let mut current = self.z.clone();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;

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
        for i in (0..m).rev() {
            let stride = 1 << i;
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
        self.z.to_vec()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.z.as_slice()
    }

    pub fn eval_as_univariate(&self, r: &F) -> F {
        let mut output = self.z[0];
        let mut rpow = *r;
        for z in self.z.iter().skip(1) {
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

        let sub_evaluations = SmartSlice::Borrowed(&self.z[start..][..chunk_size]);

        DensePolynomial {
            num_vars: self.num_vars - num_chunks.trailing_zeros() as usize,
            len: chunk_size,
            z: sub_evaluations,
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
        let num_vars_per_chunk = self.num_vars - ceil_log2(num_chunks);
        split_eval_chunks!(
            chunks_mut,
            BorrowedMut,
            &mut self.z,
            chunk_size,
            num_vars_per_chunk
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
        let num_vars_per_chunk = self.num_vars - ceil_log2(num_chunks);

        split_eval_chunks!(chunks, Borrowed, self.z, chunk_size, num_vars_per_chunk)
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
                    if i < poly.len() {
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
    use ark_std::rand::{Rng, SeedableRng};
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
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
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
                let eval2 = poly.inside_out_evaluate(&eval_point);

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
        let fixed_poly = if parallel_fix {
            poly.fix_low_parallel(&random_point[num_vars - 1])
        } else {
            poly.fix_low(&random_point[num_vars - 1])
        };

        let eval_fixed = fixed_poly.evaluate(&random_point[..num_vars - 1]).unwrap();

        assert_eq!(eval, eval_fixed);

        // fix in place
        if parallel_fix {
            poly.fix_low_mut_parallel(&random_point[num_vars - 1]);
        } else {
            poly.fix_low_mut(&random_point[num_vars - 1]);
        }

        let eval_fixed = poly.evaluate(&random_point[..num_vars - 1]).unwrap();

        assert_eq!(eval, eval_fixed);

        // test fixing high variable

        let mut poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let eval = poly.evaluate(&random_point).unwrap();
        // there is no parallel version for fixing high variable yet
        let fixed_poly = poly.new_fix_top(&random_point[0]);

        let eval_fixed = fixed_poly.evaluate(&random_point[1..]).unwrap();

        assert_eq!(eval, eval_fixed);

        // fix in place
        if parallel_fix {
            poly.par_fix_mut_top(&random_point[0]);
        } else {
            poly.fix_high_mut(&random_point[0]);
        }

        let eval_fixed = poly.evaluate(&random_point[1..]).unwrap();

        assert_eq!(eval, eval_fixed);
    }

    #[test]
    fn test_eval_endianness() {
        let num_evals = 4;
        let evals = (0..num_evals).map(Fr::from).collect();
        let poly = DensePolynomial::<Fr>::new(evals);

        let eval_at_1 = poly.evaluate(&[Fr::ZERO, Fr::ONE]).unwrap();
        let eval_at_2 = poly.evaluate(&[Fr::ONE, Fr::ZERO]).unwrap();

        assert_eq!(eval_at_1, Fr::from(1));
        assert_eq!(eval_at_2, Fr::from(2));
    }
}
