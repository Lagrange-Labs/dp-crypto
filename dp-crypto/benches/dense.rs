use ark_bn254::Fr;
use ark_poly::{Polynomial, evaluations::multivariate::DenseMultilinearExtension};
use divan::Bencher;
use dp_crypto::poly::{dense::DensePolynomial, slice::SmartSlice};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const LENS: [usize; 5] = [12, 14, 16, 18, 20];

fn arkworks_static_evals(n: usize) -> Vec<Fr> {
    (0..n).map(|i| Fr::from(i as u64)).collect()
}

#[divan::bench(args = LENS)]
fn arkyper_dense_eval_static(b: Bencher, n: usize) {
    b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
        .bench_local_refs(|s| {
            let r_len = s.len().ilog2();
            let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(s)).evaluate(&r)
        })
}

#[divan::bench(args = LENS)]
fn arkworks_dense_eval_static(b: Bencher, n: usize) {
    b.with_inputs(|| arkworks_static_evals(2u32.pow(n as u32) as usize))
        .bench_local_refs(|s| {
            let r_len = s.len().ilog2();
            let r = (0..r_len).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
            DenseMultilinearExtension::from_evaluations_slice(r_len as usize, s.as_slice())
                .evaluate(&r)
        })
}
