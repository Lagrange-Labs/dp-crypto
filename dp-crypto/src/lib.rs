// Removed unstable feature: #![feature(decl_macro)]
#![allow(dead_code)]
pub mod arkyper;
pub mod poly;
mod sumcheck;

pub use sumcheck::*;

pub mod serialization {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

    pub fn serialize<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = vec![];
        a.serialize_with_mode(&mut bytes, Compress::Yes)
            .map_err(serde::ser::Error::custom)?;
        s.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
        let a = A::deserialize_with_mode(s.as_slice(), Compress::Yes, Validate::Yes)
            .map_err(serde::de::Error::custom)?;
        Ok(a)
    }
}
