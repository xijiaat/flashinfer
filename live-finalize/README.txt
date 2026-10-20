SM90 MXFP4 live-output finalization — H20 evidence

Code: 9a913c189fc523834f2b8d7145dd76a249493aa6
Baseline: 3eb1befcf7e5ca3175c398fdf3a4569693a26bfd (BF16 amax staging already enabled in BOTH arms).
Dependency: historical #6738 bfd3072c362822c58098ddb104d6e22ebba001d3.
Hardware: four H20-3e SM90, EP4; CUDA 13.0, PyTorch 2.13.0+cu130,
CuTe DSL 4.7.1, NVSHMEM 3.8.0 / nvshmem4py 0.4.1.
This is synthetic complete-MoE LAYER validation, not model serving.

Change
Separate final combine writes only nonempty live output rows. Dispatch, peer
expert work, collective launch geometry and symmetric allocation keep their
full capacity. Empty ranks and in-kernel reduction retain the previous path.
The host launch-cache key includes output extent/stride so a pointer reused
after shrinking/growing cannot reuse the wrong reducer view.

Default-tactic complete layer
H=5120, I=2304, E=384, top_k=6, capacity=4128; all routing is synthetic.
Three rounds, each 20 alternating AB/BA pairs, five full calls per graph,
unmeasured NCCL GPU start gate; max of four rank durations. Compilation/warmup
are outside timing. The rounds share a process, not independent deployments.
Latency is median of all 60 samples/arm. Speedup = baseline/candidate - 1.
The heuristic resolves MMA(256,64,256), cluster(2,1,1), atomic_counter,
epi_warps, dispatch_warps=1, BF16 combine. No exhaustive autotuning claim.

| Tokens/rank | Routing | Baseline (us) | Candidate (us) | Speedup | Three rounds (%) |
|---:|---|---:|---:|---:|---|
| 1 | uniform | 225.914 | 193.562 | +16.714% | +16.715/+16.701/+16.763 |
| 4 | uniform | 511.997 | 479.613 | +6.752% | +6.706/+6.747/+6.796 |
| 16 | uniform | 1200.579 | 1168.646 | +2.732% | +2.726/+2.729/+2.763 |
| 128 | uniform | 1739.770 | 1711.440 | +1.655% | +1.680/+1.557/+1.666 |
| 1024 | uniform | 2666.045 | 2644.531 | +0.814% | +0.809/+0.817/+0.812 |
| 4096 | uniform | 7767.597 | 7769.370 | -0.023% | -0.063/-0.035/+0.022 |
| 16 | skew | 203.130 | 171.414 | +18.502% | +18.479/+18.559/+18.517 |
| 4 | empty-rank | 438.803 | 432.832 | +1.380% | +1.330/+1.305/+1.314 |
| 13 | ragged | 1183.232 | 1150.944 | +2.805% | +2.783/+2.840/+2.803 |

Ragged row counts are 13/14/15/16; empty-rank means rank 0 has zero rows and
the other ranks have four. M4096 is unchanged within noise, not a gain.
Largest absolute A/A round-median delta: 0.096178%.
All three cohorts, all ranks and all samples are in raw/ and tables/.
The manual K128 qualification cohort is secondary; the default cohort and
A/A use the exact final candidate hash from the receipt. Earlier manual and
smoke runs use the preserved first snapshot with equivalent device arguments.

This gain depends on reserved capacity exceeding live rows. It is mainly
useful for small decode-like batches; do not generalize the percentages to
equal-capacity buffers or model throughput. Both arms keep the same
42,270,720-byte output storage and equal post-warmup allocated/reserved bytes.
No additional weight representation or workspace reduction is claimed.

Reducer-only and NCU
M16/H5120/topk6/cap4128: unprofiled reducer ~28.65 -> 1.48 us.
NCU grid 20,640 -> 80 CTAs; profiler duration 29.760 -> 3.456 us is NOT
used in performance tables. DRAM reads 1.130496 -> 0.927744 MB; observed
DRAM writes 14.490624 -> 0 MB. Zero observed DRAM writes means the output
can remain in L2, not that the kernel writes no output. Caches/clocks were
not controlled (--clock-control none --cache-control none, nine passes).
The eliminated logical padded stores total 42,106,880 bytes in this shape.
Raw CSVs are retained; binary profiler reports remain in the local archive.

Correctness
Exact final code: official suite exit 0, 33 passed on each of four ranks;
independent numerical oracle 3 passed; staging regression 4 passed.
Shrink/grow at the same buffer address, NaN-poisoned padded output, capture
of the small view, a larger launch, then repeated small replay all pass.
New regression against the original module intentionally fails specifically
at the untouched-output-tail assertion; its numerical reference passed.
All full-layer cases verify bit-identical output in original/changed/restored
phases, with repeated graph replay. Ragged and zero-local-token ranks pass.
All 10,266 baseline files were unchanged and candidate source hashes checked.
No tolerance, precision, routing semantics or GEMM tactic changed between arms.
Applicable pre-commit checks passed for all five incremental PR files.

Preserved failures
attempt0-wrong-capture-stream: harness captured an empty graph on the wrong
stream; fixed before collecting valid reducer data.
attempt1-alias-config-identity: harness imported a duplicate Python config
class, failing isinstance before kernel execution. Fixed module identity.
negative-control: expected baseline failure, not a candidate failure.
All logs and harness versions remain under raw/. No successful sample was
filtered. A later comparison must not combine this cohort with older staging
measurements or add/multiply the two reported gains.

Reproduction
Use separate source checkouts for the baseline and final code above. The
recorded container layout is /test/source (baseline), /test/candidate-source
(candidate), /test/scripts and /test/raw; scripts/test-env.sh records paths
and build settings. Adapt dependency/venv paths to your installation.
Set PYTHONPATH to the baseline, then:

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  /test/scripts/compare_live_moe_default.py \
  --candidate /test/scripts/hopper_mxfp4_final.py \
  --out /test/raw/moe-v41-default --geometry v41 --experts 384 --default-tactic

Repeat with --control and --out /test/raw/moe-v41-aa for the A/A control.
The manual qualification script/commands are preserved in scripts/.
For final tests, set PYTHONPATH to the candidate checkout and run:
bash tests/moe_ep/run_tests.sh mega_sm90_mxfp4
python -m pytest tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py -q
python -m pytest tests/moe_ep/test_sm90_mxfp4_fused_staging.py -q
Recompute tables: python scripts/analyze.py .

Prior BF16-staging evidence is preserved at repository commit 3cd9fdef163bb5e9ee5a6d6bbc0d0358ced95125.
These experiments used an isolated test container; no production was changed.
Numerical files/scripts are unchanged; machine inventory and unrelated
environment metadata are excluded. SHA256SUMS.json covers published artifacts.
