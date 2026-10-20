"""Measure existing TopkReduce with capacity versus live output views."""
import argparse
import json
from pathlib import Path
import statistics

import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import bootstrap_paths

bootstrap_paths()
from moe_nvfp4_swapab.topk_reduce import TopkReduce
from src.token_comm import CombineFormat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ncu", choices=["baseline", "live"])
    a = p.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(62020)
    h, k, cap = 5120, 6, 4128
    data = torch.randn((cap, k, h), device="cuda", dtype=torch.bfloat16)
    ids = torch.full((cap, k), -1, device="cuda", dtype=torch.int64)
    y = torch.empty((cap, h), device="cuda", dtype=torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    op = TopkReduce(h, k, CombineFormat.parse("bf16"), sm_arch="sm_90a")
    report = {"scope": "standalone existing final reducer", "hidden": h,
              "topk": k, "capacity": cap, "cases": []}
    for m in ([16] if a.ncu else [1, 4, 16, 128, 1024, 4096, 4128]):
        ids.fill_(-1)
        ids[:m].copy_(torch.arange(k, device="cuda").expand(m, -1))
        # Also include masked live routes with poisoned combine terms.
        ids[:m:3, -1] = -1
        row = {"M": m, "checks": [], "rounds": []}
        functions = []
        for n in (cap, m):
            args = [cutlass_torch.from_dlpack(data, assumed_align=16), None,
                    cutlass_torch.from_dlpack(y[:n], assumed_align=16), None,
                    stream]
            idx = cutlass_torch.from_dlpack(ids, assumed_align=16)
            compiled = cute.compile(op, *args, topk_idx=idx)
            def call(f=compiled, args=args, idx=idx):
                # Capture uses a different current stream from eager warmup.
                args[-1] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                return f(*args, topk_idx=idx)
            functions.append(call)
        for phase in ("original", "changed", "restored"):
            if phase == "original":
                original = data.clone()
            elif phase == "changed":
                data.normal_()
            else:
                data.copy_(original)
            ref = torch.zeros((m, h), device="cuda", dtype=torch.float32)
            for route in range(k):
                ref += torch.where(ids[:m, route, None] >= 0,
                                   data[:m, route].float(), 0.0)
            for f in functions:
                f()
                torch.cuda.synchronize()
                assert torch.equal(y[:m], ref.bfloat16()), (m, phase)
            row["checks"].append(phase)
        if a.ncu:
            f = functions[0 if a.ncu == "baseline" else 1]
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            f()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            graphs = []
            for f in functions:
                for _ in range(10):
                    f()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(50):
                        f()
                graphs.append(g)
            torch.cuda.synchronize()
            for g in graphs:
                for _ in range(10):
                    g.replay()
            torch.cuda.synchronize()
            for round_id in range(3):
                samples = [[], []]
                for pair in range(20):
                    for arm in ((0, 1) if (pair + round_id) % 2 == 0 else (1, 0)):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        samples[arm].append(start.elapsed_time(end) * 1000 / 50)
                row["rounds"].append({"samples_us": samples,
                                      "medians_us": list(map(statistics.median, samples))})
            # Capture is not enough: change data and rerun both graphs.
            data.normal_()
            graphs[0].replay()
            torch.cuda.synchronize()
            expected = y[:m].clone()
            y.fill_(float("nan"))
            graphs[1].replay()
            torch.cuda.synchronize()
            assert torch.equal(y[:m], expected)
            assert torch.isnan(y[m:]).all()
            row["graph_changed_and_tail_guard"] = True
        report["cases"].append(row)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({"M": m, "round_medians": [r["medians_us"] for r in row["rounds"]]}),
              flush=True)


if __name__ == "__main__":
    main()
