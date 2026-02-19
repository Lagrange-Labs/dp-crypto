#[cfg(feature = "cuda")]
fn main() {
    use ark_bn254::{Fq, Fr};
    use ec_gpu::arkworks_bn254::G1Affine;
    use ec_gpu_gen::SourceBuilder;

    let source = SourceBuilder::new()
        .add_multiexp::<G1Affine, Fq, Fr>()
        .add_poly_ops::<Fr>();
    ec_gpu_gen::generate(&source);
}

#[cfg(not(feature = "cuda"))]
fn main() {}
