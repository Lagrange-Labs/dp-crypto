//! Example demonstrating GPU-accelerated HyperKZG polynomial commitment scheme.
//!
//! Run with:
//! ```bash
//! cargo run --example hyperkzg_gpu --features cuda
//! ```

use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use ark_std::UniformRand;
use dp_crypto::{
    arkyper::{transcript::blake3::Blake3Transcript, CommitmentScheme, HyperKZG, HyperKZGGpu},
    poly::dense::DensePolynomial,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== HyperKZG GPU Example ===\n");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

    // Parameters
    let num_vars = 16; // 2^16 = 65536 elements
    let n = 1 << num_vars;
    let num_polys = 5;

    println!("Configuration:");
    println!("  - Number of variables: {}", num_vars);
    println!("  - Polynomial size: {} elements", n);
    println!("  - Number of polynomials for batch: {}", num_polys);
    println!();

    // Generate random polynomial
    let poly_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let poly = DensePolynomial::new(poly_evals);

    // Generate random evaluation point
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

    // Setup (same for both CPU and GPU)
    println!("Setting up SRS...");
    let start = Instant::now();
    let (pk, vk) = HyperKZGGpu::<Bn254>::test_setup(&mut rng, num_vars);
    println!("  Setup time: {:?}\n", start.elapsed());

    // =========================================================================
    // Single Commit: CPU vs GPU
    // =========================================================================
    println!("--- Single Commit ---");

    // CPU commit
    let start = Instant::now();
    let (cpu_commitment, _) = HyperKZG::<Bn254>::commit(&pk, &poly)?;
    let cpu_commit_time = start.elapsed();
    println!("  CPU commit time: {:?}", cpu_commit_time);

    // GPU commit
    let start = Instant::now();
    let (gpu_commitment, _) = HyperKZGGpu::<Bn254>::commit(&pk, &poly)?;
    let gpu_commit_time = start.elapsed();
    println!("  GPU commit time: {:?}", gpu_commit_time);

    // Verify they match
    assert_eq!(cpu_commitment.0, gpu_commitment.0, "Commitments should match!");
    println!("  Commitments match!");
    println!(
        "  Speedup: {:.2}x\n",
        cpu_commit_time.as_secs_f64() / gpu_commit_time.as_secs_f64()
    );

    // =========================================================================
    // Batch Commit: CPU vs GPU
    // =========================================================================
    println!("--- Batch Commit ({} polynomials) ---", num_polys);

    // Generate multiple polynomials
    let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
        .map(|_| {
            let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            DensePolynomial::new(evals)
        })
        .collect();

    // CPU batch commit
    let start = Instant::now();
    let cpu_commitments = HyperKZG::<Bn254>::batch_commit(&pk, &polys)?;
    let cpu_batch_time = start.elapsed();
    println!("  CPU batch commit time: {:?}", cpu_batch_time);

    // GPU batch commit
    let start = Instant::now();
    let gpu_commitments = HyperKZGGpu::<Bn254>::batch_commit(&pk, &polys)?;
    let gpu_batch_time = start.elapsed();
    println!("  GPU batch commit time: {:?}", gpu_batch_time);

    // Verify they match
    for (i, ((cpu_c, _), (gpu_c, _))) in cpu_commitments.iter().zip(gpu_commitments.iter()).enumerate() {
        assert_eq!(cpu_c.0, gpu_c.0, "Commitment {} should match!", i);
    }
    println!("  All {} commitments match!", num_polys);
    println!(
        "  Speedup: {:.2}x\n",
        cpu_batch_time.as_secs_f64() / gpu_batch_time.as_secs_f64()
    );

    // =========================================================================
    // Prove (Open): CPU vs GPU
    // =========================================================================
    println!("--- Prove (Open) ---");

    // CPU prove
    let mut cpu_transcript = Blake3Transcript::new(b"example");
    let start = Instant::now();
    let cpu_proof = HyperKZG::<Bn254>::prove(&pk, &poly, &point, None, &mut cpu_transcript)?;
    let cpu_prove_time = start.elapsed();
    println!("  CPU prove time: {:?}", cpu_prove_time);

    // GPU prove
    let mut gpu_transcript = Blake3Transcript::new(b"example");
    let start = Instant::now();
    let gpu_proof = HyperKZGGpu::<Bn254>::prove(&pk, &poly, &point, None, &mut gpu_transcript)?;
    let gpu_prove_time = start.elapsed();
    println!("  GPU prove time: {:?}", gpu_prove_time);

    println!(
        "  Speedup: {:.2}x\n",
        cpu_prove_time.as_secs_f64() / gpu_prove_time.as_secs_f64()
    );

    // =========================================================================
    // Verify (same for both, runs on CPU)
    // =========================================================================
    println!("--- Verify ---");

    let eval = poly.evaluate(&point)?;

    // Verify CPU proof
    let mut verify_transcript = Blake3Transcript::new(b"example");
    let start = Instant::now();
    HyperKZG::<Bn254>::verify(
        &vk,
        &cpu_proof,
        &mut verify_transcript,
        &point,
        &eval,
        &cpu_commitment,
    )?;
    println!("  CPU proof verified in {:?}", start.elapsed());

    // Verify GPU proof
    let mut verify_transcript = Blake3Transcript::new(b"example");
    let start = Instant::now();
    HyperKZGGpu::<Bn254>::verify(
        &vk,
        &gpu_proof,
        &mut verify_transcript,
        &point,
        &eval,
        &gpu_commitment,
    )?;
    println!("  GPU proof verified in {:?}", start.elapsed());

    println!("\n=== All tests passed! ===");

    Ok(())
}
