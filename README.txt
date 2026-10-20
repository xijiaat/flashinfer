H20 SM90 BF16 amax + safe staging follow-up evidence (2026-10-20)

Baseline: flashinfer-ai/flashinfer PR #6738, bfd3072c362822c58098ddb104d6e22ebba001d3.
Candidate: the two files in candidate/ are the exact performance/NCU-tested snapshot.
All measurements are synthetic MoE-layer or staging measurements, not whole-model serving results.
The final PR adds formatting, a 64-bit launch-grid calculation, a large-index fallback guard and stronger regression inputs;
the performance-tested device-kernel body and quantization arithmetic remain AST-identical.

Contents
- tables/: all 24 complete-MoE A/B and 8 A/A rows, including every round and P10/P90.
- raw/moe-*/rank*.json: all original samples on all four ranks, including the rank-maximum
  critical latency, graph replay correctness, routing scenarios and dispatch receipts.
- raw/staging-bench/: all 42 nonempty timed shapes and six empty control shapes.
- raw/correctness/: 48 shapes x six finite-BF16 patterns; byte-level checks of all buffers.
- ncu/: all requested raw hardware metrics and kernel names for M16/M16384, both arms.
  CSV row 1 after the header contains the units; keep it when interpreting the numbers.
- raw/memory.json: incremental PyTorch CUDA allocations, not register/shared-memory usage.
- scripts/: benchmark drivers used for these results, plus a portable summary analyzer.
- candidate/: exact source used during performance/NCU measurements; SHA256.json seals files.
- final-validation/: official shell suite at final commit 3eb1befcf7e5ca3175c398fdf3a4569693a26bfd,
  33 passed on each of four ranks, 4 staging tests passed, no skips/failures; exact source hashes.

No samples were removed on the basis of observed benefit. Host identifiers, IPs, process IDs,
GPU UUIDs, unrelated environment information and NCU binary reports are not published.
The NCU CSV numeric counters are unchanged. Full original private archives are retained.

Environment: 4 x NVIDIA H20-3e, SM90, EP4; PyTorch 2.13.0+cu130, CUDA 13.0,
CUDA driver 595.71.05, CuTe DSL 4.7.1, NVSHMEM 3.8.0, nvshmem4py 0.4.1,
Python 3.12.3, Nsight Compute 2026.3.1. See environment.json.

Method
Staging: 60 alternating AB/BA pairs, 50 complete stages per graph; warmup/compilation outside timing.
Full MoE: three rounds of 20 AB/BA pairs, five full layer forwards per graph; unmeasured NCCL
start gate, then external CUDA events, cross-rank maximum reduced outside the timed region.
All rounds for a geometry run in one process, not three fresh deployments.
Includes quantization, staging, expert communication, FC1, SwiGLU, A2 quantization, FC2 and combine.
Same weights/workspace/configuration/tactic for both arms; no additional weight copy.
A/A control uses the same original stager in both arms. Capacity 4128.
Fixed qualification tactic, not a claim of optimal autotuned performance:
swap_ab=True, pingpong=False, mma=(128,32,128), cluster=(1,1,1), static,
token_back_mode=epi_warps, fast_math=True, gate_up_clamp=4, preprocess_expert_chunk_size=4.
H/I/E/K: small=4096/2048/64/4, large=7168/2048/384/8, v41=5120/2304/384/6.
V4.1 uses its captured dimensions with synthetic finite weights/routes, not a checkpoint.

Reproduce (four assigned Hopper GPUs, no other GPU work; compile time excluded)
1. In a CUDA environment, install FlashInfer at the exact baseline above with its submodules
   and the documented moe_ep NVSHMEM/CuTe dependencies. Follow FlashInfer's installation guide.
   Keep that checkout unchanged; the drivers explicitly load the separate candidate files.
2. Let BASELINE be that checkout and EVIDENCE the directory containing this README.
   export PYTHONPATH="$BASELINE:$EVIDENCE/scripts:${PYTHONPATH:-}"
   export FLASHINFER_CUDA_ARCH_LIST=9.0a FLASHINFER_JIT_DEBUG=0
   export FLASHINFER_DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=2 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1
   Set CUDA_VISIBLE_DEVICES to the four allocated GPUs. Run the commands serially.
3. python "$EVIDENCE/scripts/compare_staging.py" --candidate "$EVIDENCE/candidate" --out ./correctness --mode check
   python "$EVIDENCE/scripts/compare_staging.py" --candidate "$EVIDENCE/candidate" --out ./staging --mode bench
4. python -m torch.distributed.run --standalone --nproc-per-node=4 "$EVIDENCE/scripts/compare_complete_moe_gated_v2.py" --candidate "$EVIDENCE/candidate" --out ./moe-small --geometry small --experts 64
   Repeat with --geometry large --experts 384 --out ./moe-large,
   --geometry v41 --experts 384 --out ./moe-v41,
   and --geometry small --experts 64 --control --out ./moe-control.
   Each output directory must be fresh. A GPU gate before the events aligns rank starts.
5. NCU, one GPU, M=16 and 16384, each arm baseline/candidate:
   ncu --profile-from-start off --clock-control none --cache-control none --launch-count 20 \
     --kernel-name-base demangled --rename-kernels off \
     --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__block_size,launch__grid_size \
     --export ./report python "$EVIDENCE/scripts/compare_staging.py" --candidate "$EVIDENCE/candidate" --out ./ncu-output --mode ncu --arm candidate --tokens 16384 --hidden 7168
   CUDA_VISIBLE_DEVICES must expose only the assigned profiling GPU for this command.
   NCU used kernel replay (9 passes), no clock locking/cache flushing; its duration is
   diagnostic only. Performance claims use separate unprofiled measurements above.
6. python "$EVIDENCE/scripts/analyze.py" recomputes the complete-MoE tables from archived samples.

Scope/limits
Finite BF16 inputs only; legacy safe quantization arithmetic, tiny-value handling, scale wire,
route copying and repeated tail invalidation are preserved. Noncontiguous/unsupported paths
retain the previous behavior. No model quality claim, model throughput claim or extrapolation
to other architectures. CPU launch overhead is not included in graph replay timing.
