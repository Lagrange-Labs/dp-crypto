use std::{
    array,
    iter::Sum,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign},
};

use ark_ff::PrimeField;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    arkyper::transcript::Transcript, poly::dense::DensePolynomial, structs::IOPProverState,
    virtual_poly::VirtualPolynomial, virtual_polys::PolyMeta,
};

/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

// TODO avoid duplicate implementation with sumcheck package
/// log2 ceil of x
pub fn ceil_log2(x: usize) -> usize {
    assert!(x > 0, "ceil_log2: x must be positive");
    // Calculate the number of bits in usize
    let usize_bits = std::mem::size_of::<usize>() * 8;
    usize_bits - (x - 1).leading_zeros() as usize
}

// log2 of x; the method panics if x is not a power of two
pub fn log2_strict(x: usize) -> usize {
    assert!(x.is_power_of_two(), "log2_strict: x must be a power of two");
    x.ilog2() as usize
}

pub fn create_uninit_vec<T: Sized>(len: usize) -> Vec<MaybeUninit<T>> {
    let mut vec: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
    unsafe { vec.set_len(len) };
    vec
}

#[inline(always)]
pub fn largest_even_below(n: usize) -> usize {
    if n.is_multiple_of(2) {
        n
    } else {
        n.saturating_sub(1)
    }
}

fn prev_power_of_two(n: usize) -> usize {
    (n + 1).next_power_of_two() / 2
}

/// Largest power of two that fits the available rayon threads
pub fn max_usable_threads() -> usize {
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
    #[cfg(feature = "parallel")]
    {
        static MAX_USABLE_THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *MAX_USABLE_THREADS.get_or_init(|| {
            let n = rayon::current_num_threads();
            let threads = prev_power_of_two(n);
            if n != threads {
                tracing::warn!(
                    "thread size {n} is not power of 2, using {threads} threads instead."
                );
            }
            threads
        })
    }
}

/// transpose 2d vector without clone
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

/// merge vector of virtual poly into single virtual poly
/// NOTE this function assume polynomial in each virtual_polys are "small", due to this function need quite of clone
pub fn merge_sumcheck_polys<'a, F: PrimeField>(
    virtual_polys: Vec<&VirtualPolynomial<'a, F>>,
    poly_meta: Option<Vec<PolyMeta>>,
) -> VirtualPolynomial<'a, F> {
    assert!(!virtual_polys.is_empty());
    assert!(virtual_polys.len().is_power_of_two());
    let log2_poly_len = ceil_log2(virtual_polys.len());
    let poly_meta = poly_meta
        .unwrap_or(std::iter::repeat_n(PolyMeta::Normal, virtual_polys.len()).collect_vec());
    let mut final_poly = virtual_polys[0].clone();
    final_poly.aux_info.max_num_variables = 0;

    // usually phase1 lefted num_var is 0, thus only constant term lefted
    // but we also support phase1 stop earlier, so each poly still got num_var > 0
    // assuming sumcheck implemented in suffix alignment to batch different num_vars

    // sanity check: all PolyMeta::Normal should have the same phase1_lefted_numvar
    debug_assert!(
        virtual_polys[0]
            .flattened_ml_extensions
            .iter()
            .zip_eq(&poly_meta)
            .filter(|(_, poly_meta)| { matches!(poly_meta, PolyMeta::Normal) })
            .map(|(poly, _)| poly.num_vars())
            .all_equal()
    );
    let merged_num_vars = poly_meta
        .iter()
        .enumerate()
        .find_map(|(index, poly_meta)| {
            if matches!(poly_meta, PolyMeta::Normal) {
                let phase1_lefted_numvar =
                    virtual_polys[0].flattened_ml_extensions[index].num_vars();
                Some(phase1_lefted_numvar + log2_poly_len)
            } else {
                None
            }
        })
        .or_else(|| {
            // all poly are phase2 only, find which the max num_var
            virtual_polys[0]
                .flattened_ml_extensions
                .iter()
                .map(|poly| poly.num_vars())
                .max()
        })
        .expect("unreachable");

    final_poly.aux_info.max_num_variables =
        final_poly.aux_info.max_num_variables.max(merged_num_vars);

    final_poly
        .flattened_ml_extensions
        .par_iter_mut()
        .zip_eq(&poly_meta)
        .enumerate()
        .for_each(|(i, (ml, poly_meta))| {
            let ml_ext = match poly_meta {
                PolyMeta::Normal => DensePolynomial::new(
                    virtual_polys
                        .iter()
                        .flat_map(|virtual_poly| {
                            let mle = &virtual_poly.flattened_ml_extensions[i];
                            mle.evals()
                        })
                        .collect::<Vec<F>>(),
                ),
                PolyMeta::Phase2Only => {
                    let poly = &virtual_polys[0].flattened_ml_extensions[i];
                    assert!(poly.num_vars() <= log2_poly_len);
                    let blowup_factor = 1 << (merged_num_vars - poly.num_vars());
                    DensePolynomial::new(
                        poly.evals_ref()
                            .iter()
                            .flat_map(|e| std::iter::repeat_n(*e, blowup_factor))
                            .collect_vec(),
                    )
                }
            };
            *ml = ml_ext.into();
        });
    final_poly
}

/// retrieve virtual poly from sumcheck prover state to single virtual poly
pub fn merge_sumcheck_prover_state<'a, F: PrimeField>(
    prover_states: &[IOPProverState<'a, F>],
) -> VirtualPolynomial<'a, F> {
    merge_sumcheck_polys(
        prover_states.iter().map(|ps| &ps.poly).collect_vec(),
        Some(prover_states[0].poly_meta.clone()),
    )
}

/// we expect each thread at least take 4 num of sumcheck variables
/// return optimal num threads to run sumcheck
pub fn optimal_sumcheck_threads(num_vars: usize) -> usize {
    let expected_max_threads = max_usable_threads();
    let min_numvar_per_thread = 4;
    if num_vars <= min_numvar_per_thread {
        1
    } else {
        (1 << (num_vars - min_numvar_per_thread)).min(expected_max_threads)
    }
}

/// Derive challenge from transcript and return all power results of the challenge.
pub fn get_challenge_pows<F: PrimeField>(size: usize, transcript: &mut impl Transcript) -> Vec<F> {
    let alpha: F = transcript.append_and_sample(b"combine subset evals");

    std::iter::successors(Some(F::ONE), move |prev| Some(*prev * alpha))
        .take(size)
        .collect()
}

#[derive(Clone, Copy, Debug)]
/// util collection to support fundamental operation
pub struct AdditiveArray<F, const N: usize>(pub [F; N]);

impl<F: Default, const N: usize> Default for AdditiveArray<F, N> {
    fn default() -> Self {
        Self(array::from_fn(|_| F::default()))
    }
}

impl<F: AddAssign, const N: usize> AddAssign for AdditiveArray<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign, const N: usize> Add for AdditiveArray<F, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: AddAssign + Default, const N: usize> Sum for AdditiveArray<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}

impl<F, const N: usize> Deref for AdditiveArray<F, N> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, const N: usize> DerefMut for AdditiveArray<F, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug)]
pub struct AdditiveVec<F>(pub Vec<F>);

impl<F> Deref for AdditiveVec<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for AdditiveVec<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Clone + Default> AdditiveVec<F> {
    pub fn new(len: usize) -> Self {
        Self(vec![F::default(); len])
    }
}

impl<F: AddAssign> AddAssign for AdditiveVec<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign> Add for AdditiveVec<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: MulAssign + Copy> MulAssign<F> for AdditiveVec<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

impl<F: MulAssign + Copy> Mul<F> for AdditiveVec<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
    }
}
