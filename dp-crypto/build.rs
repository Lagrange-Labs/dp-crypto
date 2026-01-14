#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use ark_bn254::{Fq, Fq2, Fr};
    use ec_gpu::arkworks_bn254::{G1Affine, G2Affine};
    use ec_gpu_gen::SourceBuilder;

    let source = SourceBuilder::new()
        .add_multiexp::<G1Affine, Fq, Fr>()
        .add_multiexp::<G2Affine, Fq2, Fr>();
    ec_gpu_gen::generate(&source);
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}
