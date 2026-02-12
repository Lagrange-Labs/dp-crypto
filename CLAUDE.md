# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepProve Crypto Primitives - Core cryptographic library for [DeepProve](https://www.lagrange.dev/deepprove), Lagrange's verifiable AI inference engine. Provides dense multilinear polynomials, HyperKZG polynomial commitment scheme, and highly optimized sumcheck protocol implementation.

## Build Commands

```bash
# Standard build (default "mono" feature)
cargo build

# Build with parallelization
cargo build --features=all-stable

# Run tests
cargo test --features=all-stable

# Linting
cargo clippy --features=all-stable -- -D warnings

# Run benchmarks
cargo bench --bench pcs           # HyperKZG benchmarks
cargo bench --bench sumcheck      # Sumcheck benchmarks
cargo bench --bench dense         # Polynomial eval benchmarks

# GPU build (requires CUDA)
cargo build --features='parallel,cuda'
cargo test --features='parallel,cuda'
cargo run --example hyperkzg_gpu --features cuda
```

## Cargo Features

- `default = ["mono"]` - No parallelization
- `parallel` - Enables rayon parallelization and ASM optimizations
- `all-stable` - Combines mono + parallel (use for tests/CI)
- `cuda` / `opencl` - GPU acceleration via ec-gpu
- `nightly-benches` - Enables comparison benchmarks against other implementations

## Workspace Structure

```
dp-crypto/          # Main cryptographic library
sumcheck_macro/     # Procedural macro for sumcheck code generation
```

## Architecture

### Module: `arkyper` (Polynomial Commitment Scheme)
HyperKZG implementation - a KZG-based PCS for multilinear polynomials (ported from Microsoft's Nova).

- `mod.rs` - Core HyperKZG impl with `commit()`, `batch_commit()`, `open()`, `verify()`
- `hyperkzg_gpu.rs` - GPU-accelerated variant (HyperKZGGpu)
- `interface.rs` - `CommitmentScheme` trait definition
- `msm.rs` / `gpu_msm.rs` - Multi-scalar multiplication (CPU/GPU)
- `transcript/blake3.rs` - Blake3-based Fiat-Shamir transcript

### Module: `poly` (Polynomial Representations)
Dense multilinear polynomials in evaluation form.

- `dense.rs` - `DensePolynomial<F>` - main polynomial type with `evaluate()`, `fix_var()`, `linear_combination()`
- `slice.rs` - `SmartSlice` - Owned/Borrowed wrapper to avoid unnecessary copying
- `eq.rs` - Equality polynomial evaluation

### Module: `sumcheck` (Sumcheck Protocol)
Highly optimized, multi-threaded sumcheck implementation.

- `prover.rs` - `IOPProverState` with two-phase proof generation
- `verifier.rs` - `IOPVerifierState` for verification
- `virtual_polys.rs` / `virtual_poly.rs` - Polynomial expression builder
- `expression.rs` - Generic expression tree for polynomial constraints

## Key Patterns

- **Generic over Field**: All code is generic over `F: Field` or `F: PrimeField` (arkworks traits)
- **SmartSlice**: Use `SmartSlice::Borrowed` to avoid copying when possible
- **Power-of-Two Constraint**: All polynomial sizes must be powers of 2
- **Thread Count**: Use `optimal_sumcheck_threads(nv)` for parallelization (returns power of 2)
- **GPU Lazy Loading**: GPU kernels initialize lazily via `LazyLock`

## Dependencies Note

This project uses **patched arkworks** from `a16z/arkworks-algebra` branch `dev/twist-shout`. The GPU dependencies (ec-gpu, ec-gpu-gen) come from `Lagrange-Labs/ec-gpu` branch `feat/hkzg_gpu`. Be aware of these when updating dependencies.
