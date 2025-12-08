# DeepProve Crypto Primitives

This repo contains the basic primitives used for [DeepProve](https://www.lagrange.dev/deepprove), the verifiable AI inference engine built by Lagrange.
Specifically, it contains implementation of dense polynomials, PCS (HyperKZG) and sumchecks, all working over the arkworks traits ecosystem.

## Polynomials

The field is generic and is using the [arkworks trait ecosystem](https://github.com/arkworks-rs).
The polynomial can be constructed either from a vector or a slice of fields. This allows to avoid copying when unnecessary.

```rust
let poly = DensePolynomial::new_from_smart_slice(
    SmartSlice::Owned(vec![Fr::from(1),Fr::from(2),Fr::from(3),Fr::from(4)]));
let point = (0..2).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
poly.evaluate(&point);
```

## PCS

This library currently exposes [Hyperkzg](https://github.com/microsoft/Nova/blob/main/src/provider/hyperkzg.rs?ref=blog.icme.io) as a PCS:

```rust
let n = 4;
let evals = 0..2u32.pow(n as u32).map(|i| Fr::from(i)).collect::<Vec<_>>();
let (pk,vk) = HyperKZG::<Bn254>::test_setup(&mut thread_rng(), n);
let poly = DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(evals.as_slice()));
let comm = HyperKZG::<Bn254>::commit(&pp, &poly);
let point = (0..n).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();
let transcript = Blake3Transcript::new(b"hyperkzg_test");
let proof = HyperKZG::<Bn254>::open(&pk, &poly, &point, &Fr::ZERO, &mut transcript);
```

## Sumcheck

The sumcheck is a highly optimized but generic sumcheck implementation that can be used to prove over
any combination of polynomials.

```rust
let num_threads = optimal_sumcheck_threads(nv);
// we need to sample the input polynomials here because we need to provide references to
// such polynomials to the benched function
const NUM_INPUTS: usize = 3;
let poly = DensePolynomial::<F>::random(n, &mut thread_rng());

let virtual_poly = VirtualPolynomials::new_from_monomials(
    num_threads,
    nv,
    vec![Term {
        scalar: F::ONE,
        product: poly
            .iter()
            .map(Either::Left)
            .collect_vec(),
    }],
);
let transcript = T::new(b"dp_sumcheck");
IOPProverState::prove(virtual_poly, &mut transcript);
```

## Acknowledgement

The dense polynomial and HyperKZG implementation are extracted and refactored from [the Jolt codebase](https://github.com/a16z/jolt/tree/main/jolt-core).
The sumcheck implementation is ported from [the Scroll repository](https://github.com/Lagrange-Labs/gkr-backend/tree/main/crates/sumcheck).

## Licensing

dp-private is licensed under a Lagrange specific license, see [LICENSE](https://github.com/Lagrange-Labs/dp-crypto/blob/main/LICENSE) file.

## Disclaimer

This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. Lagrange Labs is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. 