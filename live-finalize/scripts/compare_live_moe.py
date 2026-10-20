"""Same public FlashInfer MoEEpLayer, weights, recipe and tactic; staging A/B only.

Run with torchrun --standalone --nproc-per-node=4. Compilation and validation
precede CUDA-event timing. No model, weight conversion or backend comparison.
"""
import argparse
import dataclasses
from datetime import timedelta
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import time

import hashlib
import importlib.util
import sys
PKG = "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)
def load_frontends(path):
    original = importlib.import_module(PKG+'.shim.hopper_mxfp4')
    name = PKG+'.shim._live_candidate'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # Share the immutable config type across alias modules; code is identical.
    module.MegaMoEHopperMxfp4Config = original.MegaMoEHopperMxfp4Config
    return original, module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--geometry', choices=['small', 'large', 'v41', 'tiny'], required=True)
    p.add_argument('--experts', type=int, default=64)
    p.add_argument('--control', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEEpTensors, Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
    from flashinfer.moe_ep.weights import PrequantizedMoEWeights
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == 4 and torch.cuda.get_device_capability() == (9, 0)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15), device_id=torch.device('cuda', rank))
    group = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
    assert dist.get_world_size() == 4
    baseline, candidate = load_frontends(a.candidate)
    public = importlib.import_module(PKG)
    baseline_launch = public.hopper_mxfp4_mega_moe
    assert baseline_launch is baseline.hopper_mxfp4_mega_moe
    observed = [0, 0]
    active = 0
    workspace = None
    def launch(*args, **kwargs):
        nonlocal workspace
        workspace = args[3]
        module = baseline if active == 0 or a.control else candidate
        workspace._frontend.__class__ = module.MegaMoEHopperMxfp4Frontend
        result = module.hopper_mxfp4_mega_moe(*args, **kwargs)
        observed[active] += 1
        n = kwargs['num_tokens']
        expected = n if active == 1 and not a.control and n > 0 else cap
        assert workspace._frontend._mega.launch_output.shape == (expected, h)
        return result
    public.hopper_mxfp4_mega_moe = launch
    h, intermediate, topk = {'small':(4096,2048,4), 'large':(7168,2048,8), 'v41':(5120,2304,6), 'tiny':(128,128,2)}[a.geometry]
    experts = a.experts
    assert experts >= topk and experts % 4 == 0
    cap = 32 if a.geometry == "tiny" else 4128
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / ('rank%d.json' % rank)
    assert not dest.exists()
    torch.manual_seed(20261018 + rank)
    # Canonical packed E2M1 + K32 E8M0. Both arms share the same transformed
    # weights and workspace; no extra weight representation is allocated.
    local_e = experts // 4
    def raw(rows, k):
        return (torch.randint(0, 256, (local_e, rows, k//2), device='cuda', dtype=torch.uint8),
                torch.randint(113, 121, (local_e, rows, k//32), device='cuda', dtype=torch.uint8))
    w1,s1 = raw(2*intermediate,h); w2,s2 = raw(h,intermediate)
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=intermediate, top_k=topk, gate_up_clamp=4.0, fast_math=True,
        preprocess_expert_chunk_size=4, swap_ab=True, pingpong=False,
        mma_tiler_mnk=(128,32,128), cluster_shape_mnk=(1,1,1),
        load_balance_mode='static', token_back_mode='epi_warps')
    report = {'status':'PREPARING', 'rank':rank, 'geometry':a.geometry, 'hidden':h,
              'intermediate':intermediate,'experts':experts,'topk':topk,'ep':4,'capacity':cap,
              'control':a.control, 'timing_protocol':'Unmeasured NCCL GPU gate, then captured external events and five complete MoE calls per graph; rank maximum', 'config':dataclasses.asdict(config), 'script_sha':sha(__file__),
              'candidate_sha':sha(a.candidate), 'baseline_sha':sha(baseline.__file__),
              'scope':'complete fused MoE including safe BF16 input quantization, staging, EP dispatch, FC1, SwiGLU, A2 quantization, FC2 and combine; fixed same upstream K128 qualification tactic, not best-autotuned claim',
              'cases':[], 'started':time.time()}
    save(dest,report)
    layer = None
    try:
        layer=MoEEpLayer(bootstrap=BootstrapConfig(world_size=4,rank=rank,process_group=dist.group.WORLD,device=rank,auto_bootstrap=True),
                        fleet_params=FleetParams(num_experts=experts,max_tokens_per_rank=cap,token_hidden_size=h),
                        weights=PrequantizedMoEWeights(w13=w1,w2=w2,w13_scale=s1,w2_scale=s2),
                        backend=MegaConfig(megakernel=config,quantize_input=True,preprocess_weights=True))
        report['allocated_after_layer']=torch.cuda.memory_allocated()
        # Include uniform and skewed routes plus a rank with no tokens. Large
        # and small graphs share one capacity; stale route tails must not leak.
        specs = ([(13,'uniform'),(1,'ragged'),(4,'empty-rank'),(32,'uniform'),(1,'uniform'),(13,'uniform')]
                 if a.smoke else [(m,'uniform') for m in (1,4,16,128,1024,4096)] + [(16,'skew'),(4,'empty-rank'),(13,'ragged')])
        for m,routing in specs:
            t=0 if routing=='empty-rank' and rank==0 else (m+rank if routing=='ragged' else m)
            x=torch.randn((t,h),device='cuda',dtype=torch.bfloat16)
            ids=torch.stack([torch.randperm(experts,device='cuda')[:topk] for _ in range(t)]) if t else torch.empty((0,topk),device='cuda',dtype=torch.int64)
            if routing=='skew': ids.copy_(torch.arange(topk,device='cuda').expand(t,-1))
            weights=torch.softmax(torch.randn((t,topk),device='cuda'),dim=-1)
            tensors=MoEEpTensors(hidden_states=x,topk_ids=ids,topk_weights=weights)
            row={'tokens_per_rank_nominal':m,'tokens_this_rank':t,'routing':routing,'status':'PREPARING','checks':[]}
            report['cases'].append(row);save(dest,report)
            active=0
            layer.warmup(tensors)
            def call(): return layer.forward(tensors,return_workspace_view=True)
            for index in (0,1):
                active=index
                for _ in range(3): call()
                torch.cuda.synchronize();dist.barrier(group=group)
            graphs=[]; outputs=[]
            for index in (0,1):
                active=index
                dist.barrier(group=group)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): y=call()
                graphs.append(g);outputs.append(y)
                torch.cuda.synchronize();dist.barrier(group=group)
            original=(x.clone(),ids.clone(),weights.clone())
            for phase in ('original','changed','restored'):
                if phase=='changed':
                    x.normal_();weights.copy_(torch.softmax(torch.randn_like(weights),dim=-1))
                    if t: ids.copy_((ids+7)%experts)
                elif phase=='restored':
                    x.copy_(original[0]);ids.copy_(original[1]);weights.copy_(original[2])
                snapshots=[]
                for index,g in enumerate(graphs):
                    for _ in range(3): g.replay()
                    torch.cuda.synchronize();snapshots.append(outputs[index].clone());dist.barrier(group=group)
                assert torch.isfinite(snapshots[0]).all() and torch.isfinite(snapshots[1]).all()
                assert torch.equal(snapshots[0],snapshots[1]), (rank,m,routing,phase,'output differs')
                row['checks'].append({'phase':phase,'replays':3,'bitwise_output':True})
            torch.cuda.synchronize();dist.barrier(group=group)
            for _ in range(10):
                for g in graphs: g.replay()
            torch.cuda.synchronize();dist.barrier(group=group)
            # Three complete independent timing rounds, alternating pair order.
            # All rank latency samples are retained; cross-rank maximum is the
            # complete operation's critical latency, reduced outside timing.
            timed_graphs=[]; events=[]; gates=[]
            for index in (0,1):
                active=index
                gate=torch.zeros(1,device='cuda',dtype=torch.int32)
                dist.all_reduce(gate);torch.cuda.synchronize();dist.barrier(group=group)
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    dist.all_reduce(gate)
                    start.record()
                    for _ in range(5):call()
                    end.record()
                timed_graphs.append(g);events.append((start,end));gates.append(gate)
                torch.cuda.synchronize();dist.barrier(group=group)
            for _ in range(10):
                for g in timed_graphs:g.replay()
            torch.cuda.synchronize();dist.barrier(group=group)
            row['rounds']=[]
            for round_id in range(1 if a.smoke else 3):
                samples=[[],[]];rank_samples=[[],[]];orders=[]
                for i in range(3 if a.smoke else 20):
                    order=(0,1) if (i+round_id)%2==0 else (1,0);orders.append(order)
                    for index in order:
                        dist.barrier(group=group)
                        start,end=events[index]
                        timed_graphs[index].replay()
                        end.synchronize()
                        elapsed=start.elapsed_time(end)*1000/5
                        values=[None]*4;dist.all_gather_object(values,elapsed,group=group)
                        samples[index].append(max(values));rank_samples[index].append(values)
                row['rounds'].append({'samples_us':samples,'all_rank_us':rank_samples,'order':orders,
                                      'median_us':[statistics.median(s) for s in samples]})
                save(dest,report)
            row['effective_tactic']=workspace._frontend.effective_tactic()
            row['weight_storage_ptrs']=[w1.data_ptr(),w2.data_ptr(),s1.data_ptr(),s2.data_ptr()]
            row['status']='PASS'
            report['memory_allocated_after_case']=torch.cuda.memory_allocated()
            save(dest,report)
            if rank==0: print(json.dumps({'m':m,'routing':routing,'round_medians':[r['median_us'] for r in row['rounds']]}),flush=True)
            for captured in graphs + timed_graphs:
                captured.reset()
            del graphs,timed_graphs,events,gates,outputs,snapshots,original,tensors,x,ids,weights
            gc.collect();torch.cuda.synchronize();dist.barrier(group=group)
        report['dispatch_calls']=observed
        assert min(observed) > 0
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        public.hopper_mxfp4_mega_moe=baseline_launch
        report['finished']=time.time();save(dest,report)
        if layer is not None and report['status']=='PASS':
            report['cleanup_phase']='before_layer_destroy';save(dest,report)
            layer.destroy()
            report['cleanup_phase']='after_layer_destroy';save(dest,report)
        if report['status']=='PASS':
            dist.destroy_process_group(group)
            dist.destroy_process_group()
            report['cleanup_phase']='complete';save(dest,report)


if __name__=='__main__':main()
