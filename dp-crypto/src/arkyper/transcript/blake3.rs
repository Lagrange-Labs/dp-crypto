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
