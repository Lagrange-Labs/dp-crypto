use super::Transcript;

pub struct Blake3Transcript {
    pub state: blake3::Hasher,
}

impl Blake3Transcript {
    pub fn new(dst: &[u8]) -> Self {
        let mut state = blake3::Hasher::new();
        state.update(dst);
        Self { state }
    }
}

impl Transcript for Blake3Transcript {
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.state.update(bytes);
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        self.state.finalize_xof().fill(out);
    }
}

#[cfg(test)]
mod test {
    use ark_ff::UniformRand;
    use ark_std::rand::thread_rng;
    use itertools::Itertools;

    use crate::arkyper::transcript::{Transcript, blake3::Blake3Transcript};
    type F = ark_bn254::Fq;
    type A = ark_bn254::G1Affine;

    #[test]
    fn test_transcript() {
        let init_transcript = || Blake3Transcript::new(b"test");
        let mut t = init_transcript();
        // generate random vector of field elements
        let scalars = (0..100).map(|_| F::rand(&mut thread_rng())).collect_vec();
        t.append_scalars(&scalars);
        // squeeze challenge
        let challenge = t.challenge_scalar::<F>();

        // re-initialize transcript and check consistency
        let mut t = init_transcript();
        scalars.iter().for_each(|s| t.append_scalars::<F>(&[s]));
        // get challenge
        let chal_2 = t.challenge_scalar();

        assert_eq!(challenge, chal_2);

        // same test but for points
        let points = (0..100).map(|_| A::rand(&mut thread_rng())).collect_vec();
        let mut t = init_transcript();
        t.append_points(&points);
        let challenge = t.challenge_scalar::<F>();

        // re-initialize transcript and check consistency
        let mut t = init_transcript();
        points.iter().for_each(|a| t.append_points::<A>(&[a]));
        // get challenge
        let chal_2 = t.challenge_scalar();

        assert_eq!(challenge, chal_2);
    }
}
