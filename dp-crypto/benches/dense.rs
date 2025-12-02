use ark_bn254::Fr;
use ark_poly::{Polynomial, evaluations::multivariate::DenseMultilinearExtension};
use divan::Bencher;
use dp_crypto::poly::{
    dense::{DensePolynomial, FixOrder},
    slice::SmartSlice,
};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const LENS: [usize; 7] = [12, 14, 16, 18, 20, 22, 24];

fn arkworks_static_evals(n: usize) -> Vec<Fr> {
    (0..n).map(|i| Fr::from(i as u64)).collect()
}

#[divan::bench_group(sample_count = 20, sample_size = 1)]
mod dense {
    use super::*;

    #[divan::bench(args = LENS)]
    fn arkyper_dense_eval(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(
                arkworks_static_evals(2u32.pow(n as u32) as usize),
            ));
            let r_len = poly.num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| poly.evaluate(point))
    }
    #[divan::bench(args = LENS)]
    fn arkyper_dense_eval_insideout(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(
                arkworks_static_evals(2u32.pow(n as u32) as usize),
            ));
            let r_len = poly.num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| poly.inside_out_evaluate(point))
    }
    #[divan::bench(args = LENS)]
    fn arkworks_dense_eval(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let evals = arkworks_static_evals(2u32.pow(n as u32) as usize);
            let r_len = evals.len().ilog2();
            let poly =
                DenseMultilinearExtension::from_evaluations_slice(r_len as usize, evals.as_slice());
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| poly.evaluate(point))
    }

    #[divan::bench(args = LENS)]
    fn arkyper_dense_dot_prod_eval_static(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(
                arkworks_static_evals(2u32.pow(n as u32) as usize),
            ));
            let r_len = poly.num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| poly.evaluate_dot_product(point))
    }

    use jolt_core::poly::dense_mlpoly::DensePolynomial as JoltDense;

    #[divan::bench(args = LENS)]
    fn jolt_dense_eval_insideout_static(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = JoltDense::new(arkworks_static_evals(2u32.pow(n as u32) as usize));
            let r_len = poly.get_num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| {
            poly.inside_out_evaluate(point);
        })
    }

    #[divan::bench(args = LENS)]
    fn jolt_dense_eval_normal(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = JoltDense::new(arkworks_static_evals(2u32.pow(n as u32) as usize));
            let r_len = poly.get_num_vars();
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| {
            poly.evaluate(point);
        })
    }
}

#[divan::bench_group(sample_count = 20, sample_size = 1)]
mod fixing {
    use super::*;
    #[divan::bench(args = LENS)]
    fn arkyper_dense_low_to_high_parallel(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(
                arkworks_static_evals(2u32.pow(n as u32) as usize),
            ));
            let r_len = poly.num_vars() / 2;
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| {
            for r_i in point {
                poly.par_fix_mut(*r_i, FixOrder::LowToHigh);
            }
        })
    }

    #[divan::bench(args = LENS)]
    fn arkyper_dense_high_to_low_parallel(b: Bencher, n: usize) {
        b.with_inputs(|| {
            let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(
                arkworks_static_evals(2u32.pow(n as u32) as usize),
            ));
            let r_len = poly.num_vars() / 2;
            let point = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            (poly, point)
        })
        .bench_local_refs(|(poly, point)| {
            for r_i in point {
                poly.par_fix_mut(*r_i, FixOrder::HighToLow);
            }
        })
    }
}
