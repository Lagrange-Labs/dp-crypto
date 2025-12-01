use ark_bn254::Fr;
use ark_poly::{Polynomial, evaluations::multivariate::DenseMultilinearExtension};
use divan::Bencher;
use dp_crypto::poly::{dense::{DensePolynomial, FixOrder}, slice::SmartSlice};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const LENS: [usize; 7] = [12, 14, 16, 18, 20,22,24];

fn arkworks_static_evals(n: usize) -> Vec<Fr> {
    (0..n).map(|i| Fr::from(i as u64)).collect()
}

#[divan::bench_group(sample_count = 3,sample_size = 1)]
mod dense {
    use super::*;

    #[divan::bench(args = LENS)]
    fn arkyper_dense_eval(b: Bencher, n: usize) {
        b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
            .bench_local_refs(|s| {
                let r_len = s.len().ilog2();
                let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
                DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(s)).evaluate(&r)
            })
    }
    #[divan::bench(args = LENS)]
    fn arkyper_dense_eval_insideout(b: Bencher, n: usize) {
        b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
            .bench_local_values(|s| {
                let r_len = s.len().ilog2();
                let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
                DensePolynomial::new_from_smart_slice(SmartSlice::Owned(s)).inside_out_evaluate(&r)
            })
    }
    #[divan::bench(args = LENS)]
    fn arkworks_dense_eval(b: Bencher, n: usize) {
        b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
            .bench_local_refs(|s| {
                let r_len = s.len().ilog2();
                let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
                DenseMultilinearExtension::from_evaluations_slice(r_len as usize, s.as_slice())
                    .evaluate(&r)
            })
    }

    use jolt_core::poly::dense_mlpoly::DensePolynomial as JoltDense;

    #[divan::bench(args = LENS)]
    fn jolt_dense_eval_insideout_static(b: Bencher, n: usize) {
        b.with_inputs(|| {
            arkworks_static_evals(2u32.pow(n as u32) as usize)
        })
        .bench_local_values(|s| {
            let n = s.len().ilog2();
            let point = (0..n).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            JoltDense::new(s).inside_out_evaluate(&point);
        })
    }

    #[divan::bench(args = LENS)]
    fn jolt_dense_eval_normal(b: Bencher, n: usize) {
        b.with_inputs(|| {
            arkworks_static_evals(2u32.pow(n as u32) as usize)
        })
        .bench_local_values(|s| {
            let n = s.len().ilog2();
            let point = (0..n).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            JoltDense::new(s).evaluate(&point);
        })
    }
}

#[divan::bench_group(sample_count = 3,sample_size = 1)]
mod fixing {
    use super::*;
    #[divan::bench(args = LENS)]
    fn arkyper_dense_low_to_high_parallel(b: Bencher, n: usize) {
        b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
        .bench_local_values(|s| {
            let r_len = s.len().ilog2()/2;
            let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            let mut poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(s));
            for r_i in r {
                poly.par_fix_mut(r_i, FixOrder::LowToHigh);
            }
        })
    }

    #[divan::bench(args = LENS)]
    fn arkyper_dense_high_to_low_parallel(b: Bencher, n: usize) {
        b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
        .bench_local_values(|s| {
            let r_len = s.len().ilog2()/2;
            let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            let mut poly = DensePolynomial::new_from_smart_slice(SmartSlice::Owned(s));
            for r_i in r {
                poly.par_fix_mut(r_i, FixOrder::HighToLow);
            }
        })
    }


}