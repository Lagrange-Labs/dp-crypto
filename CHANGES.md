# CHANGES

## 2026-02-12: Parallel CPU/GPU execution + targeted base conversion in gpu_batch_commit

### Summary
Three Rust-side-only optimizations to `gpu_batch_commit` (no GPU kernel changes):
1. **Targeted CPU base conversion**: Replace `ensure_cpu_bases_cached` (converts ALL 8M bases, ~400ms) with local conversion of only CPU-needed bases (~4096 bases, ~1ms). `ensure_cpu_bases_cached` remains for `open_gpu`.
2. **Parallel CPU/GPU via `std::thread::scope`**: GPU `batch_commit_concurrent` runs on a dedicated thread while CPU rayon MSMs run on the main thread concurrently. Saves ~478ms (CPU MSM time overlapped with GPU compute).
3. **Skip zero-fill in `batch_commit_concurrent`**: Conditional zero-fill when `poly.len() < max_len`. With power-of-2 bucketing + power-of-2 poly sizes, the fill is always skipped.

### Key design: no GPU kernel changes
Previous attempts to modify kernels (fused reduce_and_commit) caused 10x regressions. This plan only changes Rust-side orchestration.

### Files
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — `gpu_batch_commit` rewritten with targeted conversion + `std::thread::scope`
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — conditional zero-fill in `batch_commit_concurrent`

---

## 2026-02-12: REVERTED — kernel caching + fused reduce_and_commit + targeted CPU bases

Reverted all changes (kernel caching, fused kernel, targeted CPU base conversion, skip zero-fill). Benchmark showed 35.5s (10x regression from 3.4s baseline). Root cause: dispatch went from 1637ms→3523ms and sync from 1051ms→31776ms. The fused `reduce_and_commit` kernel with shared memory was still catastrophically slow despite using `num_windows` threads for Phase 1. **Lesson: stop trying to fuse these small kernels — the overhead of 3 tiny kernel launches (reduce_groups: 29 threads, reduce_windows: 1 thread, copy_at_offset: 1 thread) is negligible compared to the `multiexp_signed` kernel. The dispatch overhead is dominated by CString+cuModuleGetFunction for the MAIN kernels, not the tail kernels.**

---

## 2026-02-12: Optimize GPU batch_commit — power-of-2 bucketing, per-stream buffer reuse, configurable threshold

### Summary
Three optimizations to reduce batch_commit_concurrent overhead:
1. **Power-of-2 bucketing**: Group polys by `next_power_of_two()` instead of exact length. Reduces ~9 groups to ~4-5, cutting buffer allocations and stream sharing.
2. **Per-stream buffer reuse**: Work buffers allocated per-stream (max 4×8=32) instead of per-group (was 9×8=72). Groups sharing a stream reuse the same buffers sized to the max.
3. **Configurable GPU_MSM_THRESHOLD**: `gpu_msm_threshold()` reads `GPU_MSM_THRESHOLD` env var at runtime (default 4096), enabling threshold tuning without recompilation.

Also added timing diagnostics to `batch_commit_concurrent` (alloc/dispatch/sync/download breakdown) and `gpu_batch_commit` (bucket distribution + CPU fallback timing).

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — `batch_commit_concurrent`: per-stream buffers, timing
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — power-of-2 bucketing, env threshold, timing

---

## 2026-02-12: Multi-stream batch_commit — concurrent size group processing

### Summary
Added `batch_commit_concurrent` to process multiple size groups concurrently using separate CUDA streams. With pageable host memory, `cuMemcpyHtoDAsync` implicitly syncs the specified stream only — other streams' GPU compute continues uninterrupted during the sync wait. This enables the smaller groups' compute to run "for free" inside the largest group's sync shadows.

### Changes
- `rust-gpu-tools` CUDA: `create_stream()`, `create_kernel_on_stream()`, `write_from_buffer_on_stream()` — new multi-stream API
- `rust-gpu-tools` OpenCL: matching API using additional `CommandQueue` wrapped in `Stream` struct
- `FusedPolyCommit::batch_commit_concurrent()` — round-robin interleaved dispatch across streams, per-group work buffers, shared base buffer
- `gpu_batch_commit` now collects all GPU groups and calls `batch_commit_concurrent` in a single GPU session instead of sequential `batch_commit` per group
- Removed per-step timing diagnostics from `batch_commit`

### Files
- `ec-gpu/rust-gpu-tools/src/cuda/mod.rs` — 3 new methods on Program
- `ec-gpu/rust-gpu-tools/src/opencl/mod.rs` — 3 new methods on Program + Stream struct
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — `batch_commit_concurrent`, cleaned `batch_commit`
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — rewritten `gpu_batch_commit`

---

## 2026-02-11: Async per-poly upload in batch_commit — eliminate CPU-GPU sync overhead

### Summary
Replaced per-poly `write_from_buffer` (sync: `async_copy_from` + `stream.synchronize()`) with `write_from_buffer_async` (no sync). Since all ops run on the same CUDA stream, CUDA's in-order execution guarantees copy N completes before kernel N reads the buffer, and kernel N completes before copy N+1 overwrites it. Zero CPU-GPU syncs between polys, single sync at the end via `read_into_buffer`.

### Changes
- Added `write_from_buffer_async` to rust-gpu-tools (CUDA: drops `stream.synchronize()`, OpenCL: uses `CL_NON_BLOCKING`)
- `batch_commit` uses `write_from_buffer_async` for per-poly upload to reusable `fr_buffer`
- Buffer allocated with `unsafe { program.create_buffer(max_len) }` instead of `create_buffer_from_slice` (avoids wasted zero upload)

### Lesson (from failed bulk upload attempt)
Bulk upload (pack N polys, upload once, use `to_scalar_bytes_offset`) was SLOWER (15.6s vs 11s) because:
1. The per-poly `stream.synchronize()` wasn't wasted time — GPU computed poly N's MSM while CPU blocked on poly N+1's upload sync (free pipelining)
2. Bulk upload lost this pipelining: big upload → all kernels sequential
3. Added overhead: CPU packing (~1s), 4.8GB zero-init for packed buffer, duplicate data transfer
The real fix: make the upload async, keeping per-poly structure but eliminating the sync API call

### Files
- `ec-gpu/rust-gpu-tools/src/cuda/mod.rs` — Added `write_from_buffer_async`
- `ec-gpu/rust-gpu-tools/src/opencl/mod.rs` — Added `write_from_buffer_async`
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — `batch_commit` uses async upload

---

## 2026-02-11: Replace sort-based MSM with Pippenger in batch_commit

### Summary
Replaced the 10-kernel sort-based MSM pipeline in `batch_commit` with Pippenger MSM (`multiexp_signed` + `reduce_multiexp_groups` + `reduce_windows`). Sort-based MSM's `accumulate_sorted_buckets` step had only `total_buckets` threads (e.g. 512 for ws=4) — <1% GPU utilization on A100. Pippenger uses `work_units` threads (~27,648) — full GPU utilization. Also removed the failed `batch_commit_multi` approach which packed N polys into one sort pipeline causing massive N× memory allocation.

### Changes
- `batch_commit` now uses: `to_scalar_bytes → preprocess_signed_digits → multiexp_signed → reduce_multiexp_groups → reduce_windows → copy_at_offset` (6 async kernels per poly, 0 CPU roundtrips, single download at end)
- Removed `batch_commit_multi`, `compute_multi_batch_size`, `device_memory` field from `FusedPolyCommit`
- Removed 4 GPU kernels: `decompose_to_pairs_multi`, `prefix_sum_multi`, `accumulate_all_buckets_multi`, `reduce_windows_multi`
- `gpu_batch_commit` reverted to call `batch_commit` instead of `batch_commit_multi`
- `FusedPolyCommit::create` no longer takes `device_memory` parameter

### Lesson
Sort-based MSM's bottleneck for batch_commit is the accumulation phase: `total_buckets` threads (a few hundred to a few thousand) is far too few threads for GPU utilization. Pippenger with `work_units` threads (~27K) gives 54x more parallelism. The `batch_commit_multi` approach of packing N polys into one sort pipeline didn't help because it multiplied memory N× without fixing the underlying thread-count bottleneck.

### Files
- `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` — Removed 4 `_multi` kernels
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — Rewrote `batch_commit` (Pippenger), removed `batch_commit_multi` + `compute_multi_batch_size` + `device_memory`
- `dp-crypto/src/arkyper/hyperkzg_gpu.rs` — Reverted to `batch_commit`, updated `create()` call, updated test

---

## 2026-02-11: Concurrent multi-poly batch_commit_multi — eliminate per-poly kernel launch overhead

### Summary
New `FusedPolyCommit::batch_commit_multi` method processes N same-size polys in a single set of kernel launches instead of one-at-a-time. For 1,200 polys of size 2^17: reduces from 13,200 kernel launches + 16,800 CPU-GPU syncs to ~33 launches + 3 syncs. All polys share the same SRS bases; data is packed contiguously with bucket indices offset by poly_idx.

### Changes
- 4 new GPU kernels: `decompose_to_pairs_multi`, `prefix_sum_multi`, `accumulate_all_buckets_multi`, `reduce_windows_multi`
- Existing kernels reused unchanged at N× scale: `to_scalar_bytes`, `preprocess_signed_digits`, `count_buckets`, `scatter_to_sorted`, `reduce_buckets_chunked`, `combine_chunks_to_windows`
- `FusedPolyCommit` gains `device_memory` field for batch size computation, `compute_multi_batch_size` auto-batches to fit GPU memory
- `gpu_batch_commit` now routes GPU-sized poly groups through `batch_commit_multi` instead of sequential `batch_commit`

### Files
- `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` — 4 new kernels
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — `batch_commit_multi`, `compute_multi_batch_size`, `device_memory` field
- `dp-crypto/src/arkyper/hyperkzg_gpu.rs` — updated `gpu_batch_commit` + new `test_batch_commit_multi_correctness`

---

## 2026-02-11: batch_commit size-grouping — eliminate padding waste for mixed-size polys

### Summary
`gpu_batch_commit` previously computed `max_len` once and called `fused.batch_commit()` with all polys zero-padded to that max. For mixed-size workloads (e.g. 2,755 polys from 2^7 to 2^23), a 128-element poly ran an 8M-base MSM — ~65,536x wasted work. Now groups polys by exact length: tiny polys (≤4096 elements) run CPU MSM via rayon, GPU-sized polys get one `batch_commit` call per size group with correctly-sized bases.

### Changes
- `gpu_batch_commit`: groups polys by `len()` via BTreeMap, iterates per-group
- Tiny polys (≤ `GPU_MSM_THRESHOLD = 4096`): CPU fallback with `VariableBaseMSM::msm` parallelized via rayon
- GPU polys: one `fused.batch_commit()` per size group with `bases[..poly_len]`; persistent base buffer (uploaded once for max_len) is reused by all smaller groups
- Results written back to original indices to preserve ordering

### Files
- `dp-crypto/src/arkyper/hyperkzg_gpu.rs` — rewritten `gpu_batch_commit`, added `test_batch_commit_mixed_sizes`

---

## 2026-02-10: GPU-native SRS types — eliminate CPU↔GPU format roundtrip

### Summary
Previously both CPU and GPU HyperKZG shared the same SRS type (`HyperKZGSRS<Bn254>`), storing bases as `Vec<G1Affine>`. Every `gpu_batch_commit`/`open_gpu` call converted bases to GPU Montgomery format (`Vec<G1AffineM>`, ~50ms). New GPU-native types store bases in `Vec<G1AffineM>` directly, eliminating the conversion.

### New Types
- `HyperKZGGpuSRS`: Full SRS with `Vec<G1AffineM>` bases + verifier components. Has `trim()` and `from_cpu()`.
- `HyperKZGGpuProverKey`: Prover key with `Vec<G1AffineM>` bases. Custom serde (flat bytes). Has `from_cpu()`.

### Changes
- `CommitmentScheme for HyperKZGGpu<Bn254>`: `ProverSetup` changed from `HyperKZGProverKey<Bn254>` to `HyperKZGGpuProverKey`
- `gpu_batch_commit`: takes `&[G1AffineM]` instead of `&[G1Affine]`
- `open_gpu`: takes `&HyperKZGGpuProverKey` instead of `&HyperKZGProverKey<Bn254>`
- `gpu_setup`: returns `HyperKZGGpuSRS` instead of `HyperKZGSRS<Bn254>`, skips `powers_of_gamma_g` computation
- `GpuFusedHolder`: new `ensure_bases_uploaded_gpu(&[G1AffineM])` and `ensure_cpu_bases_cached(&[G1AffineM])` methods for direct GPU upload and lazy reverse conversion
- Added `g1affinem_to_g1affine`, `convert_bases_from_gpu` in `gpu_msm.rs` for reverse Montgomery conversion

### Files
- `dp-crypto/src/arkyper/hyperkzg_gpu.rs` — new types, updated operations, trait impl, tests
- `dp-crypto/src/arkyper/gpu_msm.rs` — reverse conversion utilities
- `dp-crypto/src/arkyper/mod.rs` — updated exports
- `dp-crypto/examples/hyperkzg_gpu.rs` — uses separate CPU/GPU prover keys

---

## 2026-02-09: Move witness polynomial computation to CPU in fused_open

### Summary
GPU witness computation (3-phase parallel synthetic division) took ~1.69s/point x 3 = 5.07s (95% of Phase 3). The algorithm is inherently sequential O(n) work with ~2.5% GPU occupancy. Moved to CPU where synthetic division runs in ~13ms/point, computed in parallel with rayon (~40ms for all 3 points). Phase 3 drops from ~5.35s to ~340ms.

### Changes
- Added `compute_witness_poly_cpu` helper function (identical to reference in `poly_ops.rs:448`)
- Phase 3 now: download combined poly to CPU (11ms) -> rayon parallel witness (40ms) -> per-point upload (11ms) + to_scalar + sort MSM
- Removed GPU witness buffers: `carries_buffer`, `propagated_carries_buffer`, `single_point_buffer`
- Removed 3-phase GPU kernel launches (phase1, carry_propagate, phase3) and their kernel name strings
- Removed `program.synchronize()` between witness and to_scalar

### Lesson
Synthetic division is inherently sequential — each element depends on the previous. GPUs need thousands of independent threads to achieve good occupancy. CPU with rayon (1 thread per eval point, 3 points) is the natural fit. Always profile whether an algorithm's parallelism structure matches the target hardware.

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — only file changed

---

## 2026-02-09: Persistent GPU bases — upload SRS once, reuse across calls

### Summary
SRS bases (~256MB for 4M points) were re-uploaded to GPU on every `batch_commit` and `fused_open` call. Added `PersistentBuffer<T>` abstraction to rust-gpu-tools that survives across `program.run()` scopes. Bases are uploaded once via `upload_bases()` and reused by all MSM operations.

### Changes
- `PersistentBuffer<T>` enum in rust-gpu-tools: wraps backend-specific `Buffer<T>`, implements `KernelArgument` on both CUDA/OpenCL, can outlive `program.run()` closures
- `FusedPolyCommit` gains `base_buffer: Option<PersistentBuffer<G::GpuRepr>>` field + `upload_bases()` method
- All 4 MSM methods (`batch_commit`, `fused_open`, `fix_vars_and_commit`, `witness_poly_batch_and_commit`) check for persistent buffer before uploading
- `GpuFusedHolder.ensure_bases_uploaded()` orchestrates CPU format conversion + GPU upload
- `open_gpu` no longer calls `.to_vec()` on bases (~110ms saved per call)
- `gpu_batch_commit` no longer calls `convert_bases_to_gpu` independently (~50ms saved)
- CUDA context management: `push_context()`/`pop_context()` for safe buffer creation/drop outside `program.run()` scope

### Files
- `ec-gpu/rust-gpu-tools/src/{lib,program,cuda/mod,opencl/mod}.rs` — PersistentBuffer type + context management
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — FusedPolyCommit persistent buffer support
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — ensure_bases_uploaded, open_gpu, gpu_batch_commit

---

## 2026-02-09: Hybrid MSM in fused_open — Pippenger Phase 2, sort-based Phase 3

### Summary
Benchmark showed Pippenger MSM is ~70x slower than sort-based for large 4M-base MSMs (Phase 3: ~1.7s/MSM vs sort's ~24.6ms/MSM). Root cause: `multiexp_signed` kernel has each thread own a 24KB private bucket array; with 27,648 threads, random bucket access causes non-coalesced global memory reads. Sort-based MSM sorts bases by bucket first, enabling sequential coalesced access.

Fix: hybrid approach — keep Pippenger for Phase 2 (small MSMs ≤2M bases, cache-friendly, 0 CPU roundtrips) and sort-based for Phase 3 (3 large ~4M-base MSMs, coalesced access). Phase 2 Pippenger buffers dropped before Phase 3 sort allocation.

### Lesson
Pippenger's per-thread-owns-buckets model scales badly on GPU for large N: bucket array per thread (2^(ws-1) × 192B) exceeds L1 cache, and random digit-based access defeats coalescing. Sort-based MSM groups accesses by bucket, enabling coalesced reads even for large N. Choose MSM algorithm based on problem size relative to cache.

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — fused_open: hybrid MSM, Phase 2 buffer lifecycle

---

## 2026-02-09: Replace sort-based MSM with Pippenger in fused_open (superseded by hybrid)

### Summary
Replaced the 14-kernel sort-based MSM pipeline in `fused_open` with Pippenger (`multiexp_signed` + `reduce_multiexp_groups` + `reduce_windows`). Sort had 3 CPU→GPU roundtrips per MSM iteration (prefix_sum sync + dispatch table download + upload). Pippenger uses 5 GPU-only kernels with 0 CPU roundtrips per MSM.

### Changes
- New `POINT_reduce_multiexp_groups` GPU kernel: sums group results per window on GPU, replacing CPU-side Horner accumulation loop
- Phase 2 + Phase 3 MSM in `fused_open`: replaced sort pipeline with `preprocess_signed_digits → multiexp_signed → reduce_multiexp_groups → reduce_windows → copy_at_offset`
- Added `eprintln!` timing instrumentation at all major phases in `fused_open` and `open_gpu`

### Lesson
Previous Pippenger attempt (2026-02-06) failed because ws=13 caused 8.5GB bucket init. Now using correct formula `min(log2(n/work_units)+2, MAX)` giving ws=9 → 1.36GB. The `multiexp_signed` kernel self-initializes buckets (line 153 in multiexp.cl), so no external fill needed.

### Files
- `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` — New `reduce_multiexp_groups` kernel
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — fused_open: Pippenger MSM + timing
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — Timing in open_gpu

---

## 2026-02-08: Pre-allocate dispatch buffers + reduce sync in fused_open

### Summary
Reverted `accumulate_all_sorted_buckets` (32.6s catastrophic regression — ~26K threads processing ~3,910 points each = terrible utilization). Restored dispatch-table approach but with pre-allocated GPU buffers and GPU-side initialization, eliminating per-iteration cuMalloc/cuMemFree overhead.

### Changes
- Pre-allocated `shared_dispatch_buffer`, `shared_partial_results_buffer`, `shared_reduce_table_buffer` once before MSM loops (eliminates 24× cuMemAlloc + cuMemFree per fused_open)
- Replaced CPU `write_from_buffer` for bucket_results/partial_results initialization with `POINT_fill_identity` GPU kernel (avoids ~39MB CPU→GPU transfer per iteration)
- Kept `u32_fill_zero` and `u32_copy_buffer` GPU kernels for counts/offsets (from previous session)
- Made `prefix_sum` use `.run()` (sync) since CPU needs dispatch table inputs; all other kernels remain `.run_async()`
- Reduced per-iteration sync points from ~20+ to 3 (prefix_sum + dispatch_table upload + reduce_table upload)

### Lesson
`accumulate_all_sorted_buckets` (1 thread per ALL buckets including empty) has catastrophic performance for BN254 at 2^22 where average bucket has ~3,910 points. Dispatch-table approach with CHUNK_SIZE=256 gives ~416K threads of ≤256 points each = much better GPU utilization.

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — fused_open method: pre-allocated dispatch buffers, GPU fill kernels, async pipeline

---

## 2026-02-08: Eliminate CPU-GPU sync bottlenecks in fused_open MSM pipeline

### Summary
Replaced dispatch-table bucket accumulation (7+ CPU-GPU syncs per MSM iteration) with `accumulate_all_sorted_buckets` kernel (1 thread per bucket, writes POINT_ZERO for empty, zero CPU transfers). All kernel calls now use `run_async()` for full pipeline overlap.

### Changes
- Replaced `accumulate_sorted_buckets`/`accumulate_chunked`/`reduce_partial_buckets` dispatch-table approach with single `accumulate_all_sorted_buckets` kernel call
- Replaced CPU `write_from_buffer` counts-zero with `u32_fill_zero` GPU kernel
- Replaced CPU `read_into_buffer`/`write_from_buffer` offsets copy with `u32_copy_buffer` GPU kernel
- Removed `write_from_buffer` pre-zero of `shared_final_result_buffer` (reduce_windows initializes acc = POINT_ZERO internally)
- Removed CPU scratch vectors: `counts_cpu`, `nonempty_ids_cpu`, `offsets_copy`, `num_nonempty_vec`, `zeros_bucket_results`
- Changed all `.run()` to `.run_async()` in both Phase 2 and Phase 3 MSM pipelines
- Removed `mut` from 4 buffer declarations no longer written from CPU

### Also
- Parallelized CPU callback polynomial evaluations with rayon (`par_iter_mut`) in `open_gpu`
- Cached `convert_bases_to_gpu` result in `GpuFusedHolder` — saves ~50ms per `open_gpu` call

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — fused_open method only
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — rayon eval, cached bases, single lock scope

---

## 2026-02-08: Fix fused_open MSM — wrong window size + wrong accumulation

### Root Cause
GPU batch_open (8.6s) was 1.8x slower than CPU (4.7s). Previous optimizations had zero effect because the actual bottleneck was two fundamental differences between fused_open and batch_commit MSM approaches:
1. **Wrong window size formula**: fused_open used `max(3, log2(n) - 8)` giving ws=13 for 2M bases → 81,920 total buckets. batch_commit uses `min(log2(n/work_units) + 2, MAX_WINDOW_SIZE)` giving ws=9 → 7,424 total buckets. **11x more work** in bucket reduction.
2. **Wrong accumulation**: fused_open used `accumulate_all_sorted_buckets_precomp` (1 thread per ALL buckets including empty). batch_commit uses dispatch-table approach: download counts, build dispatch tables on CPU, then conditional simple/chunked accumulation for non-empty buckets only.

### Changes
- Replaced all `calc_sort_window_size(n)` calls in fused_open with batch_commit's formula
- Replaced `accumulate_all_sorted_buckets_precomp` with dispatch-table accumulation (matching batch_commit)
- Removed `neg_base_buffer` + `negate_bases` kernel (saves 256MB GPU memory)
- Removed unused precomp kernels from multiexp.cl
- Switched MSM pipeline from `run_async()` to `run()` (matching batch_commit)
- Removed `calc_sort_window_size` function (no callers remain)

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — All changes
- `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` — Removed 4 precomp kernels

---

## 2026-02-06: GPU fused_open performance optimizations (4 changes)

### Summary
Applied 4 optimizations to `fused_open` to close the gap where GPU batch_open (8.55s) was 1.8x slower than CPU (4.77s). Root causes: sequential kernel execution with per-kernel synchronization, excessive cuMalloc calls, redundant small-MSM GPU overhead, and runtime point negation.

### Changes

1. **Buffer reuse** — Replaced per-iteration `create_buffer_from_slice` for coefficient and eval-point buffers with single reusable buffers + `write_from_buffer`. Reduces ~71 cuMalloc calls to ~35.

2. **Hybrid CPU/GPU for small MSMs** — Phase 2 iterations with `n_bases <= 4096` now run MSM on CPU via arkworks `VariableBaseMSM` instead of launching 14 GPU kernels. Saves ~154 kernel launches + associated cuMalloc/sync overhead for trivially small MSMs. `fused_open` signature now takes `cpu_bases` parameter.

3. **Precomputed negation kernels** — Added `POINT_accumulate_all_sorted_buckets_precomp` kernel. Precomputes negated bases once via `POINT_negate_bases`, then uses ternary select instead of runtime `FIELD_sub` in all accumulation steps. Amortized over 24 MSMs.

4. **Async kernel launches** — Vendored rust-gpu-tools into `ec-gpu/rust-gpu-tools/` with `run_async()` (launches without `stream.synchronize()`) and `synchronize()` methods. All 35 kernel calls in `fused_open` use `run_async()`. Existing `read_into_buffer` calls provide natural sync points. Eliminates ~5.6ms of per-kernel synchronization overhead and enables kernel pipelining.

### API
- `fused_open` now requires `cpu_bases: &[<G::Group as CurveGroup>::Affine]` parameter
- `G::Group` must implement `VariableBaseMSM`
- `rust-gpu-tools` is now a vendored path dependency

### Files
- `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs` — All 4 changes
- `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` — New precomp kernel
- `ec-gpu/ec-gpu-gen/Cargo.toml` — Path dep for rust-gpu-tools
- `ec-gpu/rust-gpu-tools/` — Vendored with run_async/synchronize
- `dp-crypto/dp-crypto/src/arkyper/hyperkzg_gpu.rs` — Pass cpu_bases

---

## 2026-02-06: Parallel bucket reduction for GPU MSM

### Summary
Replaced the sequential `reduce_buckets_by_window` kernel (1 thread per window, 4096-8192 sequential EC ops each) with a two-level parallel reduction: Level 1 uses shared-memory Hillis-Steele suffix sum in 256-thread blocks, Level 2 combines chunk results per window. ~10-30x speedup for the reduction phase.

### What changed
- **New `POINT_reduce_buckets_chunked` kernel**: Parallel suffix sum + tree reduction per chunk (256 buckets). Uses `cuda_shared` for shared memory.
- **New `POINT_combine_chunks_to_windows` kernel**: 1 thread/window combines chunk SBPs and chunk sums via suffix-sum + 8 doublings.
- **Removed `POINT_reduce_buckets_by_window` kernel**: Replaced by the two new kernels.
- **All 6 call sites updated**: `fix_vars_and_commit`, `witness_poly_batch_and_commit`, `batch_commit`, `fused_open` Phase 2, `fused_open` Phase 3, `multiexp_sorted`.
- **Pre-allocated chunk buffers**: `chunk_sbp` and `chunk_sum` buffers added to all pre-allocation scans.

### API
No changes to public API.

---

## 2026-02-06: Revert Pippenger, restore sort-based MSM with zero sync points in fused_open

### Summary
Pippenger's `multiexp_signed` required each thread to own a private bucket array (`work_units × 2^(ws-1)` = ~340 MB on A100), plus 8.5 GB of zero-writes for bucket initialization across 25 MSMs. Reverted to sort-based MSM but eliminated ALL CPU-GPU sync points via a new `accumulate_all_sorted_buckets` kernel that processes ALL buckets (empty ones early-exit), removing the need for `num_nonempty` download, `counts`/`nonempty_ids` download, and CPU-side dispatch table construction.

### What changed
- **New `accumulate_all_sorted_buckets` kernel**: 1 thread per bucket (total_buckets threads), checks `counts[bid] > 0` and skips empty. Replaces the download-dispatch-accumulate sequence. No `fill_identity` pre-init needed.
- **Removed `reduce_multiexp` kernel**: No longer needed — sort-based uses `reduce_buckets_by_window` + `reduce_windows`.
- **Restored `calc_sort_window_size`**: `max(3, log2(n) - 8)`, replacing `pippenger_ws`.
- **Restored sort buffers**: keys, values, sorted_values, counts, offsets, scatter_offsets, nonempty_ids, num_nonempty, bucket_results, window_results.
- **Removed Pippenger buffers**: bucket_buffer (340 MB), result_buffer (2.5 MB). ~5× less GPU memory.
- **11-step pipeline per MSM, 0 CPU syncs**: preprocess → decompose → fill_zero → count → prefix_sum → copy_offsets → scatter → accumulate_all → reduce_by_window → reduce_windows → copy_at_offset.

### API
No changes to public API.

---

## 2026-02-06: Replace sort-based MSM with Pippenger in fused_open

### Summary
Replaced the 14-step sort-based MSM pipeline in `fused_open` with Pippenger's algorithm (`multiexp_signed` + `reduce_multiexp`). Sort had 3 GPU-to-CPU downloads + dispatch table upload per MSM iteration. Pippenger uses 4 GPU-only steps with zero CPU-GPU sync points per MSM.

### What changed
- **Phase 2 + Phase 3 MSM**: Replaced sort pipeline (decompose, count, prefix_sum, copy_offsets, scatter, accumulate, reduce_buckets, reduce_windows) with: `preprocess_signed_digits` -> `multiexp_signed` -> `reduce_multiexp` -> `copy_at_offset`.
- **New `reduce_multiexp` kernel**: GPU-side Horner reduction of Pippenger group results. Replaces the two-step `reduce_buckets_by_window` + `reduce_windows`.
- **Pippenger window size**: Uses `min(log2(n/work_units) + 2, 16)` (same as non-fused `multiexp` method), replacing `log2(n) - 8`.
- **Removed sort buffers**: keys, values, sorted_values, counts, offsets, scatter_offsets, nonempty_ids, num_nonempty, bucket_results, window_results, dispatch_table, reduce_table, partial_results. Replaced with: bucket_buffer, result_buffer (plus existing digits/final_result).
- **Removed `calc_sort_window_size`**: No callers remain. Replaced by `pippenger_ws`.
- **Removed CPU scratch**: counts_cpu, nonempty_ids_cpu, dispatch_table_padded, reduce_table_padded vectors no longer needed.

### API
No changes to public API. Non-fused methods (batch_commit, fix_vars_and_commit, witness_poly_batch_and_commit) keep their sort-based approach.

---

## 2026-02-06: Fix GPU regression + optimize window size for sort-based MSM

### Summary
Reverted `accumulate_parallel` (16 threads/bucket) regression in `fused_open` — it created a huge partial_results buffer and added 15 extra Jacobian additions per bucket in the reduce step. Restored chunked dispatch (`build_dispatch_tables`) with pre-allocated GPU buffers. Also changed window size formula from `log2(n/work_units) + 2` (ws=9 for n=4M) to `log2(n) - 8` (ws=14 for n=4M), giving ~11x more parallelism and 31% fewer total field ops.

### What changed
- **Reverted accumulate_parallel/reduce_parallel**: Replaced with `fill_identity` + `build_dispatch_tables` + conditional `accumulate_sorted_buckets`/`accumulate_chunked` + `reduce_partial_buckets`. Applied to both Phase 2 and Phase 3 of fused_open.
- **Pre-allocated dispatch buffers**: dispatch_table, reduce_table, partial_results GPU buffers allocated once at max sizes during scan phase. Eliminates 75 per-iteration cuMalloc + cuMemcpyHtoD calls.
- **New window size formula**: `calc_sort_window_size(n) = max(3, log2(n) - 8)` targets ~1000 pts/bucket for sort-based MSM. Applied to all 9 call sites across gpu_buffer.rs and multiexp.rs.
- **MAX_WINDOW_SIZE raised from 10 to 16** in both gpu_buffer.rs and multiexp.rs.
- **Removed dead kernels**: `POINT_accumulate_parallel` and `POINT_reduce_parallel` from multiexp.cl.
- **CHUNK_SIZE made public** in multiexp.rs for use by gpu_buffer.rs dispatch pre-allocation.

### API
No changes to public API. FusedPolyCommit::create still accepts work_units parameter (ignored).

---

## 2026-02-05: Eliminate CPU-GPU sync points in fused_open

### Summary
Replaced CPU dispatch table construction (3 syncs/MSM × 25 MSMs = 75 syncs) with GPU-only parallel accumulation. Also batched intermediate + commitment downloads after Phase 2 loop, and used `scale_poly` to avoid uploading 128MB of zeros for combined_buffer. Total sync points reduced from ~123 to ~25.

### What changed
- **GPU-only parallel accumulation**: New `POINT_accumulate_parallel` kernel with fixed 16 threads/bucket (strided access) + `POINT_reduce_parallel` (1 thread/bucket reduces partials). Eliminates ALL dispatch-related CPU downloads (num_nonempty, counts, nonempty_ids) and CPU-side `build_dispatch_tables`.
- **Batch downloads after Phase 2 loop**: Intermediate polynomials + commitments downloaded in batch after all 22 fix_var + MSM iterations complete, instead of per-iteration. Saves 44 pipeline stalls.
- **`POINT_copy_at_offset` kernel**: Saves MSM results into GPU commitments array, enabling batch download.
- **`scale_poly` for LC init**: Replaces `create_buffer_from_slice(&vec![F::ZERO; poly_len])` (128MB upload) with uninitialized buffer + `scale_poly(poly, out, coeff, n)` which writes without reading.
- **Removed all `fill_identity` calls**: `reduce_parallel` writes identity for empty buckets, `reduce_buckets_by_window` and `reduce_windows` write unconditionally. No pre-initialization needed.
- Same GPU-only accumulation pattern applied to Phase 3 witness MSMs.

### Removed
- CPU-side dispatch vectors (`counts_cpu`, `nonempty_ids_cpu`, `num_nonempty_vec`)
- All `build_dispatch_tables` calls from fused_open
- All `POINT_fill_identity` calls from fused_open MSM pipelines

### API
No changes to public API.

---

## 2026-02-05: Restore chunked dispatch + shared buffers in fused_open

### Summary
Reverted `accumulate_all_buckets` (1-thread-per-bucket) approach — it was ~32x slower for BN254 at 2^22 where average bucket has ~8,000 points but only got 1 thread. Restored the original chunked dispatch (`build_dispatch_tables`) which splits large buckets across multiple threads.

### What changed
Combined the fast version's algorithm (commit `8917236`) with memory-efficient shared buffers:
- **Chunked dispatch restored**: Download num_nonempty + counts/nonempty_ids → `build_dispatch_tables` → conditional simple/chunked accumulation kernel. Critical for large buckets.
- **Shared pre-allocated MSM buffers**: All sort buffers (keys, values, sorted, counts, offsets, etc.) allocated once at max size, reused across all iterations. Saves ~1GB vs fresh-per-iteration.
- **GPU init kernels kept**: `u32_fill_zero` for counts reset, `POINT_fill_identity` for bucket/window/final reset. Avoids CPU→GPU transfers for buffer resets.
- **GPU offset copy kept**: `u32_copy_buffer` for scatter offsets instead of CPU roundtrip.
- **Streaming LC**: `linear_combine_accumulate` reads directly from on-GPU per-iteration buffers. No flat `polys_buffer` (~2.9GB saved).
- **Streaming witness**: One eval point at a time instead of bulk `witnesses_buffer` (~256MB saved).

### Removed
- `POINT_accumulate_all_buckets` and `POINT_accumulate_all_buckets_precomp` kernels (multiexp.cl)
- `POINT_copy_at_offset` kernel (multiexp.cl)
- `FIELD_to_scalar_bytes_max_bits` kernel (poly_ops.cl)
- Conditional neg_base_buffer precomputation

### API
No changes to public API.

---

## 2026-02-05: Fix performance regression in fused_open — GPU buffer init kernels

### Summary
The OOM fix for `fused_open` caused a performance regression due to massive CPU→GPU transfers every MSM iteration to reset buffers. Added GPU kernels to initialize buffers directly on the device, eliminating ~200MB of unnecessary transfers per `fused_open` call.

### Root Cause
The OOM fix used shared max-sized buffers for MSM operations. Since `write_from_buffer` and `read_into_buffer` require exact size match, we were:
1. Creating full-sized zero vectors on CPU (`counts_zero`, `bucket_zeros`, `window_zeros`)
2. Transferring the FULL max-sized vectors to GPU every iteration (~8MB per MSM × ~25 MSMs ≈ 200MB)

### Solution
Added two new GPU kernels to initialize buffers on-device:
- `u32_fill_zero` — initializes u32 buffers (counts) to zero
- `POINT_fill_identity` — initializes Jacobian point buffers (bucket_results, window_results, final_result) to identity

### Changes

**ec-gpu-gen/src/cl/multiexp.cl** — Added 2 new kernels:
- `u32_fill_zero(buffer, count)` — one thread per element, sets to 0
- `POINT_fill_identity(buffer, count)` — one thread per element, sets to POINT_ZERO

**ec-gpu-gen/src/gpu_buffer.rs** — Replaced `write_from_buffer` resets with kernel calls:
- Removed `counts_zero` CPU pre-allocation
- Phase 2 loop: replaced 3 `write_from_buffer` calls with kernel invocations
- Phase 3 loop: replaced 3 `write_from_buffer` calls with kernel invocations

### Expected Performance Impact
- Memory: ~1.9 GB (unchanged)
- Speed: Back to pre-OOM-fix levels (GPU kernel launch << CPU→GPU transfer)

---

## 2026-02-05: Fix OOM in fused_open — reduce GPU memory footprint

### Summary
Rewrote `fused_open` in ec-gpu-gen to reduce GPU memory usage from ~5.4GB to ~1.9GB for 22 polys at 2^22. The previous implementation allocated massive buffers (polys_buffer at 2.9GB, witnesses_buffer at 384MB) that caused OOM errors.

### Changes

**ec-gpu-gen/src/cl/poly_ops.cl** — Added `FIELD_linear_combine_accumulate` kernel:
- Streaming linear combination: `out[i] += coeff * poly[i]`
- Allows processing one polynomial at a time instead of bulk allocation

**ec-gpu-gen/src/gpu_buffer.rs** — Rewrote `fused_open` Phase 3:

1. **Streaming linear combination** (Solution 1): Replaced bulk `polys_buffer` (~2.9GB) with incremental accumulation. Each polynomial is uploaded to a reusable buffer and accumulated into `combined_buffer` one at a time.

2. **Release intermediate GPU buffers** (Solution 2): Intermediate polynomials are downloaded to CPU during Phase 2. Phase 3 now re-uploads them one at a time during streaming LC, eliminating the ~250MB of retained GPU buffers.

3. **Shared MSM sort buffers** (Solution 3): Pre-allocated ONE set of sort buffers at max(Phase2, Phase3) sizes. Reused across all MSM operations in both phases. Saves ~1GB by eliminating duplicate allocations.

4. **Streaming witness computation** (Solution 4): Instead of allocating `witnesses_buffer` for all eval points at once (~384MB), compute one witness at a time and immediately run MSM. Single reusable witness buffer (~128MB).

### Memory Budget After Fix (22 polys, 2^22)

| Buffer | Size | Notes |
|--------|------|-------|
| base_buffer | 256 MB | unchanged |
| streaming_poly_buffer | 128 MB | one poly at a time |
| combined_buffer | 128 MB | LC result |
| single_witness_buffer | 128 MB | one witness at a time |
| shared MSM buffers | ~1.25 GB | reused P2/P3 |
| **GPU Total** | **~1.9 GB** | down from ~5.4 GB |

---

## 2026-02-05: GPU-accelerated SRS (trusted setup) generation

### Summary
Added `gpu_setup()` function for GPU-accelerated SRS generation. Computing the KZG powers `[G, tau*G, tau^2*G, ..., tau^n*G]` now uses a GPU batch scalar multiplication kernel, which is significantly faster than the CPU sequential approach for large degrees.

### Changes

**ec-gpu-gen/src/cl/multiexp.cl** — Added `POINT_batch_scalar_mul` kernel:
- Windowed scalar multiplication with precomputed lookup table
- Each thread computes one `scalar[i] * base` using windowed lookup
- Table structure: `table[outer][inner] = inner * (2^(outer*window) * base)`

**ec-gpu-gen/src/gpu_buffer.rs** — Added `FusedPolyCommit::batch_scalar_mul()` method:
- Builds precomputation table on CPU (same algorithm as arkworks `BatchMulPreprocessing`)
- Uploads table + scalars to GPU
- Runs batch scalar multiplication kernel
- Returns affine points

**dp-crypto/src/arkyper/hyperkzg_gpu.rs** — Added `gpu_setup()` function:
- Same interface as `HyperKZGSRS::setup()` but uses GPU
- Computes `powers_of_beta` on CPU (fast field multiplications)
- Uses `batch_scalar_mul` for `powers_of_g` and `powers_of_gamma_g`
- G2 computations remain on CPU (small)

**dp-crypto/src/arkyper/mod.rs** — Added:
- `HyperKZGSRS::from_params()` constructor for building SRS from pre-computed params
- Export `gpu_setup` when cuda/opencl features are enabled

### Usage
```rust
use dp_crypto::arkyper::gpu_setup;

let mut rng = ark_std::test_rng();
let srs = gpu_setup(&mut rng, 1 << 22)?; // 4M powers
let (pk, vk) = srs.trim(1 << 22);
```

---

## 2026-02-05: Fix OOM in batch_commit — eliminate flat_polys buffer

### Summary
`FusedPolyCommit::batch_commit` no longer uploads all polynomial data as one massive flat GPU buffer. Previously it allocated ~2.95GB on GPU + ~2.75GB on CPU just for the polynomial data (22 polys × 2^22 × 32 bytes). Combined with other buffers, total GPU usage hit ~4.6GB causing OOM.

### Root Cause
The flat buffer existed so `to_scalar_bytes_offset` could index into it with an offset. But since we process one poly at a time in the loop anyway, we can upload each poly individually using `to_scalar_bytes` (no offset) instead.

### Changes

**ec-gpu-gen/src/gpu_buffer.rs** — Rewrote `batch_commit` loop:
- Removed: CPU-side `flat_polys: Vec<F>` that concatenated all polys (~2.75GB for 22×2^22)
- Removed: GPU-side `polys_flat_buffer` upload (~2.95GB)
- Added: reusable `padded_scratch` on CPU (size = max_len, ~128MB for 2^22)
- Added: reusable `fr_buffer` on GPU (same size)
- Per-poly loop: copy poly → padded_scratch, zero-fill tail, upload to fr_buffer, use `to_scalar_bytes` instead of `to_scalar_bytes_offset`
- Now accepts polys of varying lengths (computes max_len internally)

**dp-crypto/src/arkyper/hyperkzg_gpu.rs** — Simplified `gpu_batch_commit`:
- Removed: bulk `padded_polys: Vec<Vec<Fr>>` pre-allocation that created all padded polys at once (~2.75GB for 22×2^22)
- Now passes original poly slices directly; `batch_commit` handles padding internally per-poly

### Memory Budget After Fix (22 polys, 2^22)

| Buffer | Size | Note |
|--------|------|------|
| bases (GPU) | 256 MB | entire call |
| fr_buffer (GPU, reused) | 128 MB | one poly at a time |
| scalar_buffer (GPU, reused) | 128 MB | entire call |
| digits_buffer (GPU, reused) | 208 MB | entire call |
| keys+values+sorted_values (GPU) | 1,248 MB | entire call |
| small buffers | ~2 MB | entire call |
| **GPU Total** | **~1.97 GB** | down from ~4.6 GB |
| padded_scratch (CPU, reused) | 128 MB | one poly at a time |
| **CPU Total** | **~384 MB** | down from ~6 GB |

Net savings: **~5.6 GB** (CPU + GPU combined).

---

## 2026-02-05: Route batch_commit through sort-based MSM pipeline

### Summary
`gpu_batch_commit` now uses `FusedPolyCommit::batch_commit` (sort-based MSM) instead of `GPU_MSM.batch_msm` (old per-thread bucket MSM). This makes standalone batch commit use the same fast MSM pipeline as `fused_open`.

### Changes

**ec-gpu-gen/src/gpu_buffer.rs** — Added `FusedPolyCommit::batch_commit()` method:
- Single GPU session: uploads bases once, pre-allocates all sort buffers once (constant size since all polys are same length), then loops over polys running the full sort-based MSM pipeline per poly.
- Uses `to_scalar_bytes_offset` kernel to read from a flat packed buffer of all polynomial evaluations, doing Montgomery→standard conversion on GPU.
- Pattern matches `witness_poly_batch_and_commit` exactly (constant n_bases across iterations).

**dp-crypto/src/arkyper/hyperkzg_gpu.rs** — Changed `gpu_batch_commit` to:
- Call `GPU_FUSED.batch_commit()` instead of `GPU_MSM.batch_msm()`
- Pass Fr slices directly (no CPU `convert_scalars_to_bigint` needed — GPU does Montgomery conversion)
- Removed unused imports: `Arc`, `convert_scalars_to_bigint`, `GPU_MSM`

### What this eliminates
- CPU-side `convert_scalars_to_bigint` (Montgomery→bigint conversion moved to GPU)
- Old per-thread bucket MSM kernel (replaced by sort-based pipeline)
- `GPU_MSM` dependency for the batch_commit path

### Note
`GPU_MSM` and `gpu_msm.rs` remain for `batch_poly_msm_gpu_bn254` in `msm.rs`.

---

## 2026-02-04: Fix GPU OOM — Reduce memory usage in sort-based MSM pipeline

### Summary
Pre-allocate and reuse sort-based MSM GPU buffers across loop iterations in `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs`, and reduce CPU-side memory waste in `dp-crypto/src/arkyper/`.

### GPU buffer reuse (ec-gpu-gen/src/gpu_buffer.rs)
Hoisted 11 GPU buffer allocations out of MSM-in-a-loop patterns. Previously allocated fresh every iteration; now allocated once at max size and reused via `program.write_from_buffer()` for buffers needing re-initialization (counts, scatter_offsets, bucket_results, window_results, final_result).

**Affected methods (4 call sites):**
1. **`fix_vars_and_commit`** — n_bases halves each iteration. Max sizes computed by scanning all iterations.
2. **`witness_poly_batch_and_commit`** — Constant n_bases. Simple pre-allocation.
3. **`fused_open` Phase 2** — Same halving pattern. Buffers suffixed `_p2`.
4. **`fused_open` Phase 3** — Constant n_bases. Buffers suffixed `_p3`.

**Key detail:** Smaller n_bases → smaller window_size → MORE windows → potentially MORE total_pairs. Max buffer sizes must be computed by scanning ALL iterations, not just the first.

### CPU-side memory reduction (dp-crypto)

**hyperkzg_gpu.rs — `open_gpu` callback:**
- Removed double polynomial clone: previously `poly.evals_ref().to_vec()` then `.clone()` inside closure. Now uses a `&[Fr]` reference directly.
- Removed intermediate DensePolynomial wrapping: `eval_as_univariate` was called via `DensePolynomial::eval_as_univariate(&self)`, requiring cloning each intermediate slice into a DensePolynomial. Added standalone `eval_as_univariate(&[Fr], &Fr)` that operates on slices directly.

**gpu_msm.rs — `batch_msm`:**
- Changed signature from `&[G1AffineM]` to `Arc<Vec<G1AffineM>>`. Previously did `Arc::new(bases.to_vec())` internally, copying ~256MB of bases every call. Now caller wraps in Arc once.
- Updated all callers (hyperkzg_gpu.rs, msm.rs).

---

## 2026-02-04: Add chunked bucket accumulation and precomputed negation GPU kernels

### Summary
Added 5 new GPU kernels to `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl` (appended, no existing code modified). These kernels address two performance bottlenecks in the sort-based MSM pipeline:

1. **Load imbalance from large buckets**: The existing `POINT_accumulate_sorted_buckets` assigns 1 thread per bucket. When a bucket has ~8000 points, that thread becomes a bottleneck. The new chunked kernels split large buckets across multiple threads.

2. **Redundant runtime negation**: The existing accumulate kernel computes `FIELD_sub(FIELD_ZERO, base.y)` for every negated point in every MSM. In HyperKZG, the same bases are reused for ~25 MSMs, so precomputing negated bases once saves one FIELD_sub per negated point per MSM.

### New Kernels

1. **`POINT_accumulate_chunked`** — Phase 3b: 1 thread per (bucket, chunk) pair. Each thread accumulates a chunk of points from its portion of a bucket. Uses a CPU-built dispatch table mapping each thread to (bucket_id, chunk_start, chunk_count).

2. **`POINT_reduce_partial_buckets`** — Phase 3c: 1 thread per non-empty bucket. Tree-reduces partial results from the chunked accumulation kernel using a reduce_table mapping each bucket to its range in partial_results.

3. **`POINT_negate_bases`** — Precomputes negated affine bases: (x, y) -> (x, -y). One-time cost amortized over all MSMs in a GPU session.

4. **`POINT_accumulate_sorted_buckets_precomp`** — Optimized Phase 3: same as `POINT_accumulate_sorted_buckets` but uses ternary select between `bases` and `neg_bases` instead of runtime `FIELD_sub`.

5. **`POINT_accumulate_chunked_precomp`** — Optimized Phase 3b: chunked accumulation with precomputed negations.

### Rust-side Changes (`ec-gpu/ec-gpu-gen/src/`)

**multiexp.rs**:
- Added `CHUNK_SIZE` constant (256) — maximum points per chunk when splitting large buckets
- Added `build_dispatch_tables()` function — builds CPU-side dispatch and reduce tables for chunked accumulation. For each non-empty bucket: if count <= CHUNK_SIZE, one dispatch entry; if count > CHUNK_SIZE, split into ceil(count/CHUNK_SIZE) chunks. Returns dispatch_table (bucket_id, chunk_start, chunk_count per dispatch), reduce_table (partial_start, num_partials per bucket), and total num_dispatches.
- Updated `multiexp_sorted` method: replaced simple accumulate step with chunked dispatch. When num_dispatches == num_nonempty (no large buckets), uses the fast 1-thread-per-bucket kernel. Otherwise, runs `accumulate_chunked` + `reduce_partial_buckets`.

**gpu_buffer.rs** — Updated all 4 MSM call sites with the same chunked dispatch pattern:
1. `fix_vars_and_commit` — Phase 2 intermediate commits
2. `witness_poly_batch_and_commit` — witness MSM commits
3. `fused_open` Phase 2 — intermediate commits in fused path
4. `fused_open` Phase 3 — witness commits in fused path

Each site now:
- Downloads `counts_buffer` and `nonempty_ids_buffer` to CPU
- Calls `build_dispatch_tables()` to detect large buckets
- If no large buckets: uses existing `accumulate_sorted_buckets` kernel (fast path)
- If large buckets: uses `accumulate_chunked` then `reduce_partial_buckets` (chunked path)

**lib.rs** — Added `build_dispatch_tables` to multiexp re-exports.

### Design Decisions
- **CPU-side dispatch table construction**: The dispatch table is built on CPU because it requires scanning bucket sizes and building variable-length output. This adds two small GPU->CPU downloads (counts + nonempty_ids, ~few KB each) but avoids complex GPU-side dynamic dispatch.
- **Fast path optimization**: When no bucket exceeds CHUNK_SIZE (common for uniform scalar distributions), we skip the chunked path entirely and use the simpler 1-thread-per-bucket kernel, avoiding the overhead of dispatch/reduce table uploads.
- **CHUNK_SIZE = 256**: Chosen to balance between thread count (more chunks = more parallelism) and per-thread work (too few points per chunk = overhead dominates). May need tuning based on benchmarks.

### Notes
- All kernels placed in multiexp.cl (not ec.cl) because they use both POINT and FIELD tokens which are replaced in multiexp.cl's source generation context.
- No existing kernels were modified.
- Precomputed negation kernels (`_precomp` variants) are not yet wired into the Rust side — that will be a separate change.

---

## 2026-02-04: Sort-based MSM + XYZZ Coordinates — Full Implementation

### Summary
Implemented Changes A (sort-based bucket accumulation), B (GPU-side window accumulation), and F (XYZZ coordinates) from the "10x GPU HyperKZG Performance" plan. All MSM operations now use a sort-based pipeline with XYZZ-accelerated bucket accumulation and GPU-side Horner reduction, replacing the per-thread bucket approach.

### GPU Kernel Changes (`ec-gpu/ec-gpu-gen/src/cl/`)

**ec.cl** — Added XYZZ coordinate arithmetic (appended, no existing code modified):
- `POINT_xyzz` struct, `POINT_XYZZ_ZERO` constant
- `POINT_xyzz_add_mixed` (7M+2S vs Jacobian's 7M+4S — saves ~15% per mixed add)
- `POINT_xyzz_add` (11M+2S), `POINT_xyzz_double` (for a=0 curves like BN254)

**multiexp.cl** — Added 7 sort-based MSM kernels (appended, no existing code modified):
1. `POINT_decompose_to_pairs` — signed digits → (bucket_key, base_index) pairs
2. `POINT_count_buckets` — atomic histogram
3. `POINT_prefix_sum` — single-thread exclusive prefix sum + nonempty bucket list
4. `POINT_scatter_to_sorted` — atomic scatter to sorted order
5. `POINT_accumulate_sorted_buckets` — **uses XYZZ accumulator**, converts to Jacobian via `(X*ZZ, Y*ZZZ, ZZ)` (2M cost)
6. `POINT_reduce_buckets_by_window` — summation-by-parts per window
7. `POINT_reduce_windows` — single-thread Horner reduction (eliminates CPU accumulation)

### Rust Changes (`ec-gpu/ec-gpu-gen/src/`)

**multiexp.rs**:
- Added `multiexp_sorted` method to `SingleMultiexpKernel` (full 13-step sort pipeline)
- Added `SortedMsmParams` struct and `compute_sorted_msm_params` function
- Added `calc_chunk_size_sorted` for sort-based memory budgeting
- Fixed 3 bugs: count_buckets/scatter_to_sorted passed wrong args (2 u32s instead of 1 `total_pairs`); decompose global work size was `n_bases` instead of `n_bases * num_windows`

**gpu_buffer.rs** — Replaced all 4 MSM call sites:
1. `fix_vars_and_commit` — Phase 2 intermediate commits
2. `witness_poly_batch_and_commit` — witness MSM commits
3. `fused_open` Phase 2 — intermediate commits in fused path
4. `fused_open` Phase 3 — witness commits in fused path
- Removed: pre-allocated bucket/result buffers, `num_groups`, CPU Horner accumulation loops, `use std::ops::AddAssign`
- Each MSM now downloads 1 point (96 bytes) instead of `work_units` points (~1.5MB)

**lib.rs** — Added `SortedMsmParams`, `compute_sorted_msm_params` to exports

### Key Technical Details

**Sort pipeline per MSM** (after `preprocess_signed_digits`):
```
decompose → count (atomics) → prefix_sum (1 thread) → download num_nonempty →
copy offsets → scatter (atomics) → accumulate (XYZZ, 1 thread/bucket) →
reduce_by_window (1 thread/window) → reduce_windows (Horner, 1 thread) → download 1 point
```

**XYZZ → Jacobian conversion** (no field inversion needed):
- `(X_jac, Y_jac, Z_jac) = (X * ZZ, Y * ZZZ, ZZ)` — just 2 multiplications per bucket

**Memory**: Sort-based approach uses LESS total GPU memory than per-thread buckets (no `work_units * bucket_len` allocation), despite needing sorting buffers.

### Lessons Learned
- GPU kernel args are positional — passing 2 u32s where 1 is expected causes silent corruption
- `num_windows = div_ceil(effective_bits + 1, window_size)` — signed-digit carry needs +1
- `total_pairs = n_bases * num_windows` — decompose/count/scatter thread count must match
- XYZZ→Jacobian conversion is cheap (2M) if you use `Z_jac = ZZ` instead of computing Z = ZZZ/ZZ

### Status
- Change A (sort-based bucket accumulation): **COMPLETED**
- Change B (GPU-side window accumulation): **COMPLETED**
- Change C (precomputed negation kernels): **COMPLETED** — GPU kernels added, Rust wiring pending
- Change D (large bucket splitting): **COMPLETED** — chunked dispatch with CPU-side dispatch tables
- Change E (double-buffered chunk processing): **DEFERRED** — requires rust-gpu-tools multi-stream API
- Change F (XYZZ coordinates): **COMPLETED** — used in accumulate kernels

---

## 2026-02-03: GPU HyperKZG open — Five-Change Performance Optimization Pass

### Problem
GPU `open` (7.25s) was **13% slower** than CPU `open` (6.42s) at 2^22 poly size despite GPU batch_commit being roughly equal (~3s each). Goal: make GPU significantly faster.

### Changes

#### Change 3: Skip unnecessary poly.evaluate() in GPU prove (~100-200ms savings)
- **File**: `dp-crypto/src/arkyper/hyperkzg_gpu.rs`
- `prove()` was calling `poly.evaluate()` but the result was ignored by `open_gpu` (it recomputes evaluations internally). Changed to pass `Fr::ZERO` directly, skipping the expensive O(n) evaluation.

#### Change 4: Optimize MSM bit extraction in field.cl (~100-300ms across all MSMs)
- **File**: `ec-gpu/ec-gpu-gen/src/cl/field.cl`
- Replaced loop-based `FIELD_get_bits_lsb` (10 iterations per call, called 100M+ times per MSM) with direct bit manipulation: 1-2 shift+mask operations per call.
- Extracts bits by indexing the correct limb directly (`l.val[limb_idx] >> bit_idx`) and handles cross-limb boundaries with a single conditional OR.

#### Change 1: Parallel witness polynomial computation (~200-280ms savings)
- **File**: `ec-gpu/ec-gpu-gen/src/cl/poly_ops.cl`, `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs`
- The `FIELD_witness_poly_batch` kernel launched **only 3 threads** (one per evaluation point), each doing 4M sequential iterations. Replaced with a 3-phase parallel approach:
  - **Phase 1** (`FIELD_witness_poly_batch_phase1`): 4096 threads per eval point each process a chunk independently (assuming carry_in = 0)
  - **Phase 2** (`FIELD_witness_carry_propagate`): 1 thread per eval point propagates carries right-to-left across chunks
  - **Phase 3** (`FIELD_witness_poly_batch_phase3`): 4096 threads per eval point apply carry corrections
- Updated all 3 call sites in gpu_buffer.rs: `fused_open`, `witness_poly_batch_and_commit`, `eval_and_witness_batch`.

#### Change 2: Eliminate padded_polys CPU→GPU upload (~175-375ms savings)
- **Files**: `ec-gpu/ec-gpu-gen/src/cl/poly_ops.cl`, `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs`, `dp-crypto/src/arkyper/hyperkzg_gpu.rs`
- Previously: callback created 22 polynomials zero-padded to 4M length each, then uploaded them as a 2.8GB flat buffer. The intermediate GPU buffers from Phase 1+2 were already on GPU but got dropped.
- Added `FIELD_copy_and_pad` GPU kernel to copy+zero-fill polynomial data into a flat padded buffer on GPU.
- Restructured `fused_open` to keep all intermediate `fr_out_buffer`s alive in a Vec instead of overwriting.
- After callback returns, builds flat padded buffer on GPU using `copy_and_pad` kernel — no CPU→GPU transfer of padded data.
- Removed `padded_polys` field from `Phase3Input` struct. Updated `open_gpu` callback accordingly.

#### Change 5: Signed-digit bucket decomposition for MSM (~300-600ms savings)
- **Files**: `ec-gpu/ec-gpu-gen/src/cl/multiexp.cl`, `ec-gpu/ec-gpu-gen/src/multiexp.rs`, `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs`
- Converts unsigned bucket indices to signed (Booth encoding), halving bucket count from `2^w - 1` to `2^(w-1)`.
- New `POINT_preprocess_signed_digits` kernel: one thread per base, scans through all windows and computes signed digits with carry propagation.
- New `POINT_multiexp_signed` kernel: same algorithm as original but with half the buckets. Point negation via `y = FIELD_sub(FIELD_ZERO, y)` for short Weierstrass curves.
- Updated `SingleMultiexpKernel::multiexp` and `calc_chunk_size` in multiexp.rs.
- Updated all 4 MSM call sites in gpu_buffer.rs to use signed-digit decomposition.

### Bugs Found & Fixed
- **Critical: parallel witness carry propagation** — The initial `FIELD_witness_carry_propagate` kernel had the add/multiply order wrong and was missing an extra factor of `u`. The carry recurrence is: `carry_in = carries[c-1] + carry_in_prev * u^chunk_size`, and the propagated carry needs `* u` because the rightmost element's correction is `carry_in * u`. Fixed in poly_ops.cl.
- **Critical: FIELD_sub/FIELD_ZERO undefined in multiexp.cl** — The `POINT_multiexp_signed` kernel uses `FIELD_sub(FIELD_ZERO, base.y)` for point negation, but `source.rs` only replaced `POINT` and `EXPONENT` tokens in `multiexp.cl`, not `FIELD`. The base field `FIELD` token was only replaced in `ec.cl`. Fixed by adding `.replace("FIELD", &F::name())` to the multiexp source generation in `source.rs`. **Lesson: when adding code to a `.cl` template that uses tokens from another template's namespace, check `source.rs` to verify all tokens are replaced.**
- **Critical: signed-digit carry overflow losing MSBs** — `num_windows = div_ceil(effective_bits, window_size)` is correct for unsigned representation but NOT for signed-digit (Booth) encoding. Signed digits can produce a carry OUT of the last window. Example: scalar `3` with `window_size=2` → window 0 gets `val=3 >= half=2`, produces digit `-1` with `carry=1` to window 1. But if `effective_bits=2`, `num_windows=1`, so the carry is lost and the scalar is represented as `-1` instead of `3`. For BN254 with 254-bit scalars and `window_size=2`: `ceil(254/2)=127` windows cover bits 0-253, but a carry from window 126 would need window 127. Fixed by using `div_ceil(effective_bits + 1, window_size)` everywhere. **Lesson: signed-digit representation of an N-bit value can require N+1 bits.**

### Expected Performance Impact
| Optimization | Savings |
|-------------|---------|
| Skip unnecessary eval | 100-200ms |
| Fast bit extraction | 100-300ms |
| Parallel witness poly | 200-280ms |
| Eliminate padded_polys upload | 175-375ms |
| Signed-digit buckets | 300-600ms |
| **Total** | **875-1755ms** |

Target: GPU 7.25s → ~5.5-6.4s (beating CPU 6.42s).

### Verification Needed
- `cargo test --features='parallel,cuda'` — all GPU vs CPU comparison tests must pass
- `cargo bench --bench cpu_vs_gpu --features cuda` — target: GPU open < 6.0s
- `RUST_LOG=debug cargo run --example hyperkzg_gpu --features cuda` — verify per-phase timing

---

## 2026-02-03: Fused GPU Operations for HyperKZG open_gpu

### Problem
The previous `open_gpu` implementation used multiple separate GPU sessions:
1. One session for fix_vars (Phase 1) via `GPU_POLY_OPS`
2. One session for intermediate commits (Phase 2) via `GPU_MSM.batch_msm`
3. Multiple sessions for Phase 3 (linear_combine, witness_poly_batch, witness MSMs) each using separate GPU sessions

This caused:
- Bases uploaded to GPU twice (once for Phase 2 commits, once for Phase 3 witness commits)
- Witness polynomials downloaded from GPU after computation, then scalars converted on CPU, then re-uploaded for MSM
- Linear combination result downloaded then re-uploaded for witness computation
- Scalar conversion (`convert_scalars_to_bigint`) running on CPU instead of GPU

### Changes

#### Step 1: Add `compute_work_units` to ec-gpu-gen
- **File**: `ec-gpu/ec-gpu-gen/src/multiexp.rs`
- Exposed the internal `work_units()` calculation as a public function `compute_work_units(device: &Device) -> usize`.
- Re-exported from `ec-gpu-gen/src/lib.rs`.
- Needed by `FusedPolyCommit::create()` which requires work_units as a parameter.

#### Step 2: Add `fused_open` to `FusedPolyCommit`
- **File**: `ec-gpu/ec-gpu-gen/src/gpu_buffer.rs`
- New structs: `Phase3Input<F, G>` (callback return type), `FusedOpenResult<F, AffineG, ProjectiveG>` (final result).
- New method `FusedPolyCommit::fused_open()` runs the entire HyperKZG open in a single `program_closures!` scope:
  - Phase 1+2: fix_var iterations + to_scalar_bytes ON GPU + MSM commits, all reusing one base_buffer
  - CPU callback: transcript work (append commitments, get challenges, eval polynomials, compute q_powers)
  - Phase 3: linear_combine + witness_poly_batch + to_scalar_bytes_offset + MSM commits, reusing the SAME base_buffer
- **Key technique**: `middle_fn` callback is passed as the `arg` parameter to `program.run()`, which avoids the `FnOnce` + `program_closures!` macro double-closure issue (when both cuda+opencl features are enabled, the macro creates two closures from the body, but `arg` is only passed to the chosen one at runtime).

#### Step 3: Add `GPU_FUSED` lazy static and rewrite `open_gpu`
- **File**: `dp-crypto/src/arkyper/hyperkzg_gpu.rs`
- New `GpuFusedHolder` struct with lazy initialization, exposed as `GPU_FUSED` static.
- Rewrote `open_gpu` to use `GPU_FUSED.fused_open()` with a closure that handles all CPU transcript work.
- The closure receives intermediates + commitments from Phase 1+2, does transcript/eval/challenge work, and returns `Phase3Input` with padded polys + coefficients + eval points.

#### Step 4: Remove now-unused functions
- **File**: `dp-crypto/src/arkyper/hyperkzg_gpu.rs`
- Removed `kzg_open_batch_gpu` — replaced by fused Phase 3 inside `fused_open`.
- Removed `gpu_batch_commit_with_bases` — was only used by `open_gpu` and `gpu_batch_commit`.
- Made `gpu_batch_commit` self-contained (no longer delegates to `gpu_batch_commit_with_bases`).
- Removed `Arc` import (no longer needed).
- Kept `gpu_fix_var`, `gpu_fix_vars_with_intermediates`, `gpu_linear_combine`, `gpu_witness_poly`, `gpu_witness_poly_batch` — still used by tests and potentially external callers.

### Key Decisions
- **Callback pattern for transcript work**: The `middle_fn` callback runs CPU-side transcript operations while GPU buffers remain alive. This is safe because `FnOnce` is only called once, and all GPU buffers from Phase 1+2 (especially `base_buffer`) persist through the callback into Phase 3.
- **Using `program.run(closures, middle_fn)`**: The `arg` parameter to `program.run()` avoids the double-move problem with `program_closures!` macro when both cuda+opencl features are enabled.
- **`FusedOpenResult` has 3 type params**: `AffineG` for pass-through affine commitments from the callback, `ProjectiveG` for MSM outputs. This avoids requiring conversion traits on the generic types.

### Data Flow Comparison

**Before (multiple GPU sessions):**
```
↑ upload poly + challenges              ← CPU→GPU (session 1: fix_vars)
↓ download intermediates                ← GPU→CPU
↑ upload bases + scalars                ← CPU→GPU (session 2: batch_commit)
↓ download commitments                  ← GPU→CPU
CPU: transcript → r, u, q_powers
↑ upload padded polys                   ← CPU→GPU (session 3: linear_combine)
↓ download combined poly                ← GPU→CPU
↑ upload combined + points              ← CPU→GPU (session 4: witness_poly_batch)
↓ download witness polys                ← GPU→CPU
CPU: convert_scalars_to_bigint          ← CPU work
↑ upload bases (AGAIN!) + scalars       ← CPU→GPU (session 5: batch_msm)
↓ download witness commitments          ← GPU→CPU
```

**After (single GPU session):**
```
↑ upload poly + challenges + bases ONCE ← CPU→GPU (single session)
GPU: fix_var iterations
GPU: to_scalar_bytes ON GPU
GPU: MSM commits (reuse base_buffer)
↓ download intermediates + MSM results  ← GPU→CPU (needed for transcript)
CPU: transcript → r, u, q_powers       (callback, GPU buffers alive)
↑ upload padded polys + coeffs + points ← CPU→GPU (small relative to bases)
GPU: linear_combine → stays on GPU
GPU: witness_poly_batch → stays on GPU
GPU: to_scalar_bytes_offset ON GPU
GPU: MSM commits (REUSE base_buffer!)
↓ download witness MSM results          ← GPU→CPU
```

### Verification
- Existing tests (`test_open_gpu_vs_cpu`, `test_hyperkzg_gpu_trait_vs_cpu_trait`) verify GPU and CPU paths produce identical proofs.
- `cargo test --features='parallel,cuda'` — all GPU vs CPU comparison tests must pass.
- `RUST_LOG=debug cargo run --example hyperkzg_gpu --features cuda` — verify timing improvements.

---

## 2026-02-03: GPU HyperKZG Performance Optimization

### Problem
GPU `open()` was slower than necessary due to:
1. Redundant GPU memory transfers: bases were converted to GPU format independently for Phase 2 (intermediate commitments) and Phase 3 (KZG batch open witness MSMs) — two separate `convert_bases_to_gpu` calls for the same data.
2. Redundant bases copies: `bases_gpu[..msm_size].to_vec()` allocated new vectors per-polynomial in `gpu_batch_commit` and `batch_poly_msm_gpu_bn254`, even though `multiexp`'s internal `skip` parameter handles slicing.
3. Phase 3 (`kzg_open_batch`) ran entirely on CPU: linear combination, witness polynomial computation, and witness MSMs all used CPU despite GPU kernels being available.

### Changes

#### Step 1: Eliminate bases copy in `gpu_batch_commit` and `batch_poly_msm_gpu_bn254`
- **Files**: `hyperkzg_gpu.rs`, `msm.rs`
- Removed `Arc::new(bases_gpu[..msm_size].to_vec())` — passes the full `bases_gpu` Arc instead.
- `multiexp` internally does `bases_arc[0..exps.len()]` when `skip=0`, so only the correct number of bases are used.
- Saves one large allocation + copy per polynomial in batch operations.

#### Step 2: Create `kzg_open_batch_gpu`
- **File**: `hyperkzg_gpu.rs`
- New function that replaces CPU `kzg_open_batch` for the GPU path.
- Uses `gpu_linear_combine` instead of `DensePolynomial::linear_combination` for computing the batched polynomial. Pads shorter polys with zeros (GPU kernel requires same-length inputs).
- Uses `gpu_witness_poly_batch` instead of per-point `compute_witness_polynomial` for witness polynomials — single GPU call for all 3 points.
- Reuses pre-converted bases Arc for witness MSMs — no redundant base conversion.
- Added `gpu_witness_poly_batch` wrapper for the ec-gpu-gen batch witness kernel.

#### Step 3: Refactor `open_gpu` to share bases across all phases
- **File**: `hyperkzg_gpu.rs`
- Bases are converted to GPU format once at the start of `open_gpu`.
- The same Arc is passed to Phase 2 (`gpu_batch_commit_with_bases`) and Phase 3 (`kzg_open_batch_gpu`).
- New `gpu_batch_commit_with_bases` function accepts pre-converted bases Arc.
- Previously: 2 base conversions (one in `gpu_batch_commit`, one in `batch_poly_msm_gpu_bn254` inside `kzg_open_batch`). Now: 1 base conversion total.

#### Step 4: Add tracing instrumentation
- **File**: `hyperkzg_gpu.rs`
- Added `tracing::debug_span!` to all phases of `open_gpu`: fix_vars, convert_bases, batch_commit_intermediates, kzg_open_batch.
- Added spans inside `kzg_open_batch_gpu`: linear_combine, witness_polys, witness_msm.
- Added spans to `gpu_batch_commit` and `gpu_batch_commit_with_bases`.
- Use `RUST_LOG=debug` to see per-operation timings.

#### Step 5: Add MSM scalar bitlength benchmark
- **File**: `benches/msm_bitlength.rs` (new)
- Benchmarks GPU and CPU MSM with 53-bit vs 256-bit scalars at sizes 2^14, 2^16, 2^18.
- Purpose: verify that ec-gpu-gen's `compute_max_scalar_bits()` optimization is working. If 53-bit scalars aren't significantly faster, the optimization needs investigation.

#### Step 6: Switch all batch MSM to `batch_multiexp` (eliminates per-call GPU base upload)
- **Files**: `hyperkzg_gpu.rs`, `msm.rs`, `gpu_msm.rs`
- **Critical finding**: `multiexp` re-uploads bases to GPU via `create_buffer_from_slice` on every call, even when the same `Arc<Vec<G1AffineM>>` is passed. The previous Arc sharing only avoided CPU-side conversion, NOT the PCIe transfer.
- Added `GpuMsm::batch_msm` wrapper around ec-gpu-gen's `batch_multiexp`, which uploads bases to GPU once and runs the MSM kernel per scalar set.
- Rewrote all batch MSM call sites: `gpu_batch_commit`, `gpu_batch_commit_with_bases`, `kzg_open_batch_gpu` witness MSMs, and `batch_poly_msm_gpu_bn254`.
- `batch_multiexp` requires all scalar sets to have the same length as bases — shorter polys are zero-padded.

#### Step 7: Cleanup pass
- **Files**: `gpu_msm.rs`, `hyperkzg_gpu.rs`, `msm.rs`
- **Removed duplicate `PolyOpsKernel`**: `GpuMsm` previously created both an MSM kernel and a `PolyOpsKernel`. The poly_ops methods on `GpuMsm` (`fix_var`, `fix_vars`, `linear_combine`, `witness_poly`) were dead code — all callers use `GPU_POLY_OPS` from `hyperkzg_gpu.rs`. Removed the `poly_ops` field and all wrapper methods from `GpuMsm`, saving GPU resources (one fewer kernel initialization).
- **Optimized `batch_poly_msm_gpu_bn254`**: Previously converted all `g1_powers` to GPU format regardless of actual polynomial lengths. Now computes `max_poly_len` and only converts `bases[..max_poly_len]`, avoiding unnecessary base conversion and GPU upload.
- **Deduplicated `gpu_batch_commit`**: Now delegates to `gpu_batch_commit_with_bases` instead of duplicating its logic.
- **Avoided unnecessary allocation in `kzg_open_batch_gpu` linear combine**: The largest polynomial (size n) was being copied via `.to_vec()` even though it didn't need padding. Now uses `Option<Vec<Fr>>` to reference the original slice directly for max-length polys, avoiding a 32MB+ allocation for large polynomials.

### Key Decisions
- **Zero-padding for `gpu_linear_combine`**: GPU kernel requires all polys to have the same length. Since the intermediate polynomials have decreasing sizes (n, n/2, ..., 2), we pad shorter ones with zeros. This is mathematically equivalent to the CPU path which skips out-of-range indices.
- **Witness poly length difference**: GPU `witness_poly_batch` returns length `n-1` (omits trailing zero). CPU `compute_witness_polynomial` returns length `n` (trailing zero included). The MSM result is identical because zero * base = identity. This means GPU witness MSMs use `n-1` bases vs CPU's `n` bases — slightly less work.

### Verification
- Existing tests (`test_open_gpu_vs_cpu`, `test_hyperkzg_gpu_trait_vs_cpu_trait`) verify that GPU and CPU paths produce identical proofs.
- `cargo test --features='parallel,cuda'` — all GPU vs CPU comparison tests must pass.
- `RUST_LOG=debug cargo run --example hyperkzg_gpu --features cuda` — verify tracing output.
- `cargo bench --bench msm_bitlength --features cuda` — verify scalar bitlength optimization.
