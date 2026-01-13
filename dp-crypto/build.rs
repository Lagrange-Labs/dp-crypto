#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use ark_bn254::{Fq, Fr, G1Affine};
    use ec_gpu_gen::SourceBuilder;

    let source = SourceBuilder::new().add_multiexp::<G1Affine, Fq, Fr>();
    ec_gpu_gen::generate(&source);
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}
