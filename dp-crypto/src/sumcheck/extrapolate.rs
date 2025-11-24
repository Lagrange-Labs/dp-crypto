use ark_ff::Field;
use itertools::Itertools;
use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    marker::PhantomData,
    sync::{Arc, Mutex, OnceLock},
};

/// Precomputed extrapolation weights using the second form of barycentric interpolation.
///
/// This table supports extrapolation of univariate polynomials where:
/// - The known values are at integer points `x = 0, 1, ..., d`
/// - The degree `d` is in a fixed range [`min_degree`, `max_degree`]
/// - A univariate polynomial of degree `d` has exactly `d + 1` evaluation points
/// - The extrapolated values are computed at integer points `z > d`, up to `max_degree`
/// - No field inversions are required at runtime
///
/// The second form of the barycentric interpolation formula is:
///
/// ```text
/// L(z) = ∑_{j=0}^d (w_j / (z - x_j)) / ∑_{j=0}^d (w_j / (z - x_j)) * f(x_j)
///      = ∑_{j=0}^d v_j * f(x_j)
/// ```
///
/// Where:
/// - `x_j = j` (fixed integer evaluation points)
/// - `w_j = 1 / ∏_{i ≠ j} (x_j - x_i)` are barycentric weights (precomputed)
/// - `v_j = (w_j / (z - x_j)) / denom` are normalized interpolation coefficients (precomputed)
///
/// This structure stores all `v_j` coefficients for each `(degree, target_z)` pair.
/// At runtime, extrapolation is done by a simple dot product of `v_j` with the known values `f(x_j)`,
/// without needing any inverses.
pub struct ExtrapolationTable<F: Field> {
    /// weights[degree][z - degree - 1][j] = coefficient for f(x_j) when extrapolating to z
    pub weights: Vec<Vec<Vec<F>>>,
}

impl<F: Field> ExtrapolationTable<F> {
    pub fn new(min_degree: usize, max_degree: usize) -> Self {
        let mut weights = Vec::new();

        for d in min_degree..=max_degree {
            let mut degree_weights = Vec::new();

            let xs: Vec<F> = (0..=d as u64).map(F::from).collect_vec();
            let mut bary_weights = Vec::new();

            // Compute barycentric weights w_j = 1 / prod_{i != j} (x_j - x_i)
            for j in 0..=d {
                let mut w = F::ONE;
                for i in 0..=d {
                    if i != j {
                        w *= xs[j] - xs[i];
                    }
                }
                bary_weights.push(w.inverse().unwrap()); // safe because all x_i are distinct
            }

            for z_idx in d + 1..=max_degree {
                let z = F::from(z_idx as u64);
                let mut den = F::ZERO;
                let mut tmp: Vec<F> = Vec::with_capacity(d + 1);

                for j in 0..=d {
                    let t = bary_weights[j] / (z - xs[j]);
                    tmp.push(t);
                    den += t;
                }

                // Normalize
                for t in tmp.iter_mut() {
                    *t /= den;
                }

                degree_weights.push(tmp);
            }

            weights.push(degree_weights);
        }

        Self { weights }
    }
}

pub struct ExtrapolationCache<F> {
    _marker: PhantomData<F>,
}

impl<F: Field> ExtrapolationCache<F> {
    fn global_cache() -> &'static Mutex<BTreeMap<TypeId, Box<dyn Any + Send + Sync>>> {
        static GLOBAL_CACHE: OnceLock<Mutex<BTreeMap<TypeId, Box<dyn Any + Send + Sync>>>> =
            OnceLock::new();
        GLOBAL_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    #[allow(clippy::type_complexity)]
    fn cache_map() -> Arc<Mutex<BTreeMap<(usize, usize), Arc<ExtrapolationTable<F>>>>> {
        let global = Self::global_cache();
        let mut map = global.lock().unwrap();

        map.entry(TypeId::of::<F>())
            .or_insert_with(|| {
                Box::new(Arc::new(Mutex::new(BTreeMap::<
                    (usize, usize),
                    Arc<ExtrapolationTable<F>>,
                >::new()))) as Box<dyn Any + Send + Sync>
            })
            .downcast_ref::<Arc<Mutex<BTreeMap<(usize, usize), Arc<ExtrapolationTable<F>>>>>>()
            .expect("TypeId mapped to wrong type")
            .clone()
    }

    /// precompute and cache `ExtrapolationTable`s for all `(min_degree, max_degree)`
    /// pairs where `2 ≤ max_degree` and `1 ≤ min_degree < max_degree`.
    pub fn warm_up(max_degree: usize) {
        assert!(max_degree >= 2, "max_degree must be at least 2");

        for max in 2..=max_degree {
            for min in 1..max {
                let _ = Self::get(min, max);
            }
        }
    }

    /// get or create a cached `ExtrapolationTable` for the range `(min_degree, max_degree)`.
    pub fn get(min_degree: usize, max_degree: usize) -> Arc<ExtrapolationTable<F>> {
        let cache = Self::cache_map();
        let mut map = cache.lock().unwrap();

        if let Some(existing) = map.get(&(min_degree, max_degree)) {
            return existing.clone();
        }

        let table = Arc::new(ExtrapolationTable::new(min_degree, max_degree));
        map.insert((min_degree, max_degree), table.clone());
        table
    }
}

/// extrapolates values of a univariate polynomial in-place using precomputed barycentric weights.
///
/// this function fills in the remaining entries of `uni_variate[start..]` assuming the first `start`
/// values are evaluations of a univariate polynomial at `0, 1, ..., start - 1`.
/// it uses a precomputed [`ExtrapolationTable`] from [`ExtrapolationCache`] to perform
/// efficient barycentric extrapolation without requiring any inverse operations at runtime.
///
/// Note: this function is highly optimized without field inverse. see [`ExtrapolationTable`] for how to achieve it
pub fn extrapolate_from_table<F: Field>(uni_variate: &mut [F], start: usize) {
    let cur_degree = start - 1;
    let table = ExtrapolationCache::<F>::get(cur_degree, uni_variate.len() - 1);
    let target_len = uni_variate.len();
    assert!(start > 0, "start must be > 0 to define a degree");
    assert!(
        target_len > start,
        "no extrapolation needed if target_len <= start"
    );

    let (known, to_extrapolate) = uni_variate.split_at_mut(start);
    let weight_sets = &table.weights[0]; // since min_degree == cur_degree

    for (offset, target) in to_extrapolate.iter_mut().enumerate() {
        let weights = &weight_sets[offset];
        assert_eq!(weights.len(), known.len());

        let acc = weights
            .iter()
            .zip(known.iter())
            .fold(F::ZERO, |acc, (w, x)| acc + (*w * *x));

        *target = acc;
    }
}

fn extrapolate_uni_poly_deg_1<F: Field>(p_i: &[F; 2], eval_at: F) -> F {
    let x0 = F::ZERO;
    let x1 = F::ONE;

    // w0 = 1 / (0−1) = -1
    // w1 = 1 / (1−0) =  1
    let w0 = -F::ONE;
    let w1 = F::ONE;

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;

    let l = d0 * d1;
    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;

    l * (t0 + t1)
}

fn extrapolate_uni_poly_deg_2<F: Field>(p_i: &[F; 3], eval_at: F) -> F {
    let x0 = F::from(0);
    let x1 = F::from(1);
    let x2 = F::from(2);

    // w0 = 1 / ((0−1)(0−2)) =  1/2
    // w1 = 1 / ((1−0)(1−2)) = -1
    // w2 = 1 / ((2−0)(2−1)) =  1/2
    let w0 = F::from(1).div(F::from(2));
    let w1 = -F::ONE;
    let w2 = F::from(1).div(F::from(2));

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;
    let d2 = eval_at - x2;

    let l = d0 * d1 * d2;

    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();
    let inv_d2 = d2.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;
    let t2 = w2 * p_i[2] * inv_d2;

    l * (t0 + t1 + t2)
}

fn extrapolate_uni_poly_deg_3<F: Field>(p_i: &[F; 4], eval_at: F) -> F {
    let x0 = F::from(0);
    let x1 = F::from(1);
    let x2 = F::from(2);
    let x3 = F::from(3);

    // w0 = 1 / ((0−1)(0−2)(0−3)) = -1/6
    // w1 = 1 / ((1−0)(1−2)(1−3)) =  1/2
    // w2 = 1 / ((2−0)(2−1)(2−3)) = -1/2
    // w3 = 1 / ((3−0)(3−1)(3−2)) =  1/6
    let w0 = -F::from(1).div(F::from(6));
    let w1 = F::from(1).div(F::from(2));
    let w2 = -F::from(1).div(F::from(2));
    let w3 = F::from(1).div(F::from(6));

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;
    let d2 = eval_at - x2;
    let d3 = eval_at - x3;

    let l = d0 * d1 * d2 * d3;

    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();
    let inv_d2 = d2.inverse().unwrap();
    let inv_d3 = d3.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;
    let t2 = w2 * p_i[2] * inv_d2;
    let t3 = w3 * p_i[3] * inv_d3;

    l * (t0 + t1 + t2 + t3)
}

fn extrapolate_uni_poly_deg_4<F: Field>(p_i: &[F; 5], eval_at: F) -> F {
    let x0 = F::from(0);
    let x1 = F::from(1);
    let x2 = F::from(2);
    let x3 = F::from(3);
    let x4 = F::from(4);

    // w0 = 1 / ((0−1)(0−2)(0−3)(0−4)) =  1/24
    // w1 = 1 / ((1−0)(1−2)(1−3)(1−4)) = -1/6
    // w2 = 1 / ((2−0)(2−1)(2−3)(2−4)) =  1/4
    // w3 = 1 / ((3−0)(3−1)(3−2)(3−4)) = -1/6
    // w4 = 1 / ((4−0)(4−1)(4−2)(4−3)) =  1/24
    let w0 = F::from(1).div(F::from(24));
    let w1 = -F::from(1).div(F::from(6));
    let w2 = F::from(1).div(F::from(4));
    let w3 = -F::from(1).div(F::from(6));
    let w4 = F::from(1).div(F::from(24));

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;
    let d2 = eval_at - x2;
    let d3 = eval_at - x3;
    let d4 = eval_at - x4;

    let l = d0 * d1 * d2 * d3 * d4;

    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();
    let inv_d2 = d2.inverse().unwrap();
    let inv_d3 = d3.inverse().unwrap();
    let inv_d4 = d4.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;
    let t2 = w2 * p_i[2] * inv_d2;
    let t3 = w3 * p_i[3] * inv_d3;
    let t4 = w4 * p_i[4] * inv_d4;

    l * (t0 + t1 + t2 + t3 + t4)
}

fn extrapolate_uni_poly_deg_5<F: Field>(p_i: &[F; 6], eval_at: F) -> F {
    let x0 = F::from(0);
    let x1 = F::from(1);
    let x2 = F::from(2);
    let x3 = F::from(3);
    let x4 = F::from(4);
    let x5 = F::from(5);

    // w0 = 1 / ((0−1)(0−2)(0−3)(0−4)(0-5)) =  -1/120
    // w1 = 1 / ((1−0)(1−2)(1−3)(1−4)(1-5)) = 1/24
    // w2 = 1 / ((2−0)(2−1)(2−3)(2−4)(2-5)) =  -1/12
    // w3 = 1 / ((3−0)(3−1)(3−2)(3−4)(3-5)) = 1/12
    // w4 = 1 / ((4−0)(4−1)(4−2)(4−3)(4-5)) =  -1/24
    // w5 = 1 / ((5−0)(5−1)(5−2)(5−3)(5-4)) =  1/120
    let w0 = -F::from(1).div(F::from(120));
    let w1 = -w0 * x5;
    let w2 = -w1 * x2;
    let w3 = -w2;
    let w4 = -w1;
    let w5 = -w0;

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;
    let d2 = eval_at - x2;
    let d3 = eval_at - x3;
    let d4 = eval_at - x4;
    let d5 = eval_at - x5;

    let l = d0 * d1 * d2 * d3 * d4 * d5;

    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();
    let inv_d2 = d2.inverse().unwrap();
    let inv_d3 = d3.inverse().unwrap();
    let inv_d4 = d4.inverse().unwrap();
    let inv_d5 = d5.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;
    let t2 = w2 * p_i[2] * inv_d2;
    let t3 = w3 * p_i[3] * inv_d3;
    let t4 = w4 * p_i[4] * inv_d4;
    let t5 = w5 * p_i[5] * inv_d5;

    l * (t0 + t1 + t2 + t3 + t4 + t5)
}

fn extrapolate_uni_poly_deg_6<F: Field>(p_i: &[F; 7], eval_at: F) -> F {
    let x0 = F::from(0);
    let x1 = F::from(1);
    let x2 = F::from(2);
    let x3 = F::from(3);
    let x4 = F::from(4);
    let x5 = F::from(5);
    let x6 = F::from(6);

    // w0 = 1 / ((0−1)(0−2)(0−3)(0−4)(0-5)(0-6)) =  1/720
    // w1 = 1 / ((1−0)(1−2)(1−3)(1−4)(1-5)(1-6)) = -1/120
    // w2 = 1 / ((2−0)(2−1)(2−3)(2−4)(2-5)(2-6)) = 1/48
    // w3 = 1 / ((3−0)(3−1)(3−2)(3−4)(3-5)(3-6)) = -1/36
    // w4 = 1 / ((4−0)(4−1)(4−2)(4−3)(4-5)(4-6)) =  1/48
    // w5 = 1 / ((5−0)(5−1)(5−2)(5−3)(5-4)(5-6)) =  -1/120
    // w6 = 1 / ((6−0)(6−1)(6−2)(6−3)(6-4)(6-5)) =  1/720
    let w0 = F::from(1).div(F::from(720));
    let w1 = -w0 * x6;
    let w2 = F::from(1).div(F::from(48));
    let w3 = -F::from(1).div(F::from(36));
    let w4 = w2;
    let w5 = w1;
    let w6 = w0;

    let d0 = eval_at - x0;
    let d1 = eval_at - x1;
    let d2 = eval_at - x2;
    let d3 = eval_at - x3;
    let d4 = eval_at - x4;
    let d5 = eval_at - x5;
    let d6 = eval_at - x6;

    let l = d0 * d1 * d2 * d3 * d4 * d5 * d6;

    let inv_d0 = d0.inverse().unwrap();
    let inv_d1 = d1.inverse().unwrap();
    let inv_d2 = d2.inverse().unwrap();
    let inv_d3 = d3.inverse().unwrap();
    let inv_d4 = d4.inverse().unwrap();
    let inv_d5 = d5.inverse().unwrap();
    let inv_d6 = d6.inverse().unwrap();

    let t0 = w0 * p_i[0] * inv_d0;
    let t1 = w1 * p_i[1] * inv_d1;
    let t2 = w2 * p_i[2] * inv_d2;
    let t3 = w3 * p_i[3] * inv_d3;
    let t4 = w4 * p_i[4] * inv_d4;
    let t5 = w5 * p_i[5] * inv_d5;
    let t6 = w6 * p_i[6] * inv_d6;

    l * (t0 + t1 + t2 + t3 + t4 + t5 + t6)
}

/// Evaluate a univariate polynomial defined by its values `p_i` at integer points `0..p_i.len()-1`
/// using Barycentric interpolation at the given `eval_at` point.
///
/// For overall idea, refer to https://people.maths.ox.ac.uk/trefethen/barycentric.pdf formula 3.3
/// barycentric weights `w` are for polynomial interpolation.
/// for a fixed set of interpolation points {x_0, x_1, ..., x_n}, the barycentric weight w_j is defined as:
/// w_j = 1 / ∏_{k ≠ j} (x_j - x_k)
/// these weights are used in the barycentric form of Lagrange interpolation, which allows
/// for efficient evaluation of the interpolating polynomial at any other point
/// the weights depend only on the interpolation nodes and can be treat as `constant` in loop-unroll + inline version
///
/// This is a runtime-dispatched implementation optimized for small degrees
/// with unrolled loops for performance
///
/// # Arguments
/// * `p_i` - Values of the polynomial at consecutive integer points.
/// * `eval_at` - The point at which to evaluate the interpolated polynomial.
///
/// # Returns
/// The value of the polynomial `eval_at`.
pub fn extrapolate_uni_poly<F: Field>(p: &[F], eval_at: F) -> F {
    match p.len() {
        2 => extrapolate_uni_poly_deg_1(p.try_into().unwrap(), eval_at),
        3 => extrapolate_uni_poly_deg_2(p.try_into().unwrap(), eval_at),
        4 => extrapolate_uni_poly_deg_3(p.try_into().unwrap(), eval_at),
        5 => extrapolate_uni_poly_deg_4(p.try_into().unwrap(), eval_at),
        6 => extrapolate_uni_poly_deg_5(p.try_into().unwrap(), eval_at),
        7 => extrapolate_uni_poly_deg_6(p.try_into().unwrap(), eval_at),
        _ => unimplemented!("Extrapolation for degree {} not implemented", p.len() - 1),
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;

    #[test]
    fn test_extrapolate_from_table() {
        type F = ark_bn254::Fq;
        fn f(x: u64) -> F {
            F::from(2u64) * F::from(x) + F::from(3u64)
        }
        // Test a known linear polynomial: f(x) = 2x + 3

        let degree = 1;
        let target_len = 5; // Extrapolate up to x=4

        // Known values at x=0 and x=1
        let mut values: Vec<F> = (0..=degree as u64).map(f).collect();

        // Allocate extra space for extrapolated values
        values.resize(target_len, F::ZERO);

        // Run extrapolation
        extrapolate_from_table(&mut values, degree + 1);

        // Verify values against f(x)
        for (x, val) in values.iter().enumerate() {
            let expected = f(x as u64);
            assert_eq!(*val, expected, "Mismatch at x={}", x);
        }
    }
}
