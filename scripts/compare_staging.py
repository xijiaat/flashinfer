"""Same-recipe staging A/B; original package stays unchanged."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time
import types

PKG = 'flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_stagers(candidate):
    baseline = importlib.import_module(PKG + '.staging')
    name = PKG + '._fusion_candidate_safe'
    spec = importlib.util.spec_from_file_location(name, candidate / 'safe_quantize.py')
    safe = importlib.util.module_from_spec(spec)
    sys.modules[name] = safe
    spec.loader.exec_module(safe)
    module = types.ModuleType(PKG + '._fusion_candidate_stage')
    module.__package__ = PKG
    module.__file__ = str(candidate / 'staging.py')
    source = Path(module.__file__).read_text()
    source = source.replace('from .safe_quantize import', 'from ._fusion_candidate_safe import')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return baseline.stage_mega_moe_inputs, module.stage_mega_moe_inputs


def save(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def buffers(torch, t, h, k, cap):
    return [torch.empty((cap, h), device='cuda', dtype=torch.float8_e4m3fn),
            torch.empty((cap, 4), device='cuda', dtype=torch.float32),
            torch.empty((cap, k), device='cuda', dtype=torch.int64),
            torch.empty((cap, k), device='cuda', dtype=torch.float32)]


def poison(torch, outputs):
    for tensor in outputs:
        tensor.view(torch.uint8).fill_(0x5A)


def compare(torch, a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), 'buffer ' + str(i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['smoke', 'check', 'bench', 'ncu'], required=True)
    p.add_argument('--arm', choices=['baseline', 'candidate'], default='baseline')
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--hidden', type=int, default=5120)
    a = p.parse_args()
    import torch
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_capability() == (9, 0)
    baseline, candidate = load_stagers(a.candidate)
    a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'mode': a.mode, 'cases': [],
              'script_sha': sha(__file__), 'candidate_sha': {f: sha(a.candidate / f) for f in ['staging.py','safe_quantize.py']},
              'baseline_sha': {f: sha(Path(importlib.import_module(PKG+'.staging').__file__).parent/f) for f in ['staging.py','safe_quantize.py']},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'scope': 'Same recipe, same buffers; staging only, not complete MoE or model throughput'}
    random.seed(20261018)
    torch.manual_seed(20261018)
    if a.mode == 'smoke':
        shapes = [(6, 5120, 6, 32), (1, 129, 2, 513)]
    elif a.mode == 'check':
        shapes = [(t,h,k,max(t+32,64)) for t in (0,1,4,16,128,1024,4096,16384) for h,k in ((1,1),(65,2),(129,2),(4096,4),(5120,6),(7168,8))]
    elif a.mode == 'bench':
        shapes = [(t,h,(2,6,8)[(i+j)%3],t+tail) for i,t in enumerate((0,1,4,16,128,1024,4096,16384)) for j,h in enumerate((4096,5120,7168)) for tail in (0,32)]
    else:
        shapes = [(a.tokens, a.hidden, 6, a.tokens+32)]
    try:
        for t,h,k,cap in shapes:
            x = torch.randn((t,h), device='cuda', dtype=torch.bfloat16)
            ids = torch.randint(-1,64,(t,k),device='cuda',dtype=torch.int64)
            weights = torch.randn((t,k),device='cuda',dtype=torch.float32)
            dest = [buffers(torch,t,h,k,cap) for _ in range(2)]
            def call(fn, outputs):
                fn(x,weights,ids,*outputs,quantize_input=True,safe_quantization=True)
            row = {'tokens':t,'hidden':h,'topk':k,'capacity':cap,'checks':[], 'status':'RUNNING'}
            report['cases'].append(row)
            # The original function is loaded from unchanged upstream files.
            patterns = ('normal','zero','tiny','outlier','subnormal','finite_bits') if a.mode=='check' else ('normal',)
            for pattern in patterns:
                if pattern=='zero': x.zero_()
                elif pattern=='tiny': x.normal_().mul_(1e-32)
                elif pattern=='outlier':
                    x.normal_()
                    if t:
                        x[:,0] = -128
                        x[:,-1] = 1e-38
                elif pattern=='subnormal':
                    x.fill_(torch.finfo(torch.bfloat16).tiny / 128)
                    if t: x[:,0] = -torch.finfo(torch.bfloat16).tiny
                elif pattern=='finite_bits':
                    # Exercise all finite BF16 exponent ranges, including tiny-safe branch.
                    bits=torch.randint(0,65536,x.shape,device='cuda',dtype=torch.int32)
                    bits[(bits & 0x7f80)==0x7f80]=0
                    x.copy_(bits.to(torch.int16).view(torch.bfloat16))
                for fn,d in zip((baseline,candidate),dest):
                    poison(torch,d)
                    call(fn,d)
                torch.cuda.synchronize()
                compare(torch,*dest)
                row['checks'].append({'pattern':pattern,'bitwise_all_buffers':True})
            if a.mode in ('smoke','check') and t:
                graphs=[]
                for fn,d in zip((baseline,candidate),dest):
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): call(fn,d)
                    graphs.append(g)
                original=(x.clone(),ids.clone(),weights.clone())
                for phase in ('original','changed','restored'):
                    if phase=='changed':
                        x.normal_();ids.random_(-1,64);weights.normal_()
                    if phase=='restored':
                        x.copy_(original[0]);ids.copy_(original[1]);weights.copy_(original[2])
                    for _ in range(3):
                        for g,d in zip(graphs,dest): poison(torch,d);g.replay()
                        torch.cuda.synchronize();compare(torch,*dest)
                    row['checks'].append({'phase':phase,'replays':3,'bitwise_all_buffers':True})
                del graphs
            if a.mode=='bench' and t:
                graphs=[]
                for fn in (baseline,candidate):
                    for _ in range(10): call(fn,dest[0])
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(50): call(fn,dest[0])
                    graphs.append(g)
                torch.cuda.synchronize()
                for _ in range(10):
                    for g in graphs:g.replay()
                torch.cuda.synchronize()
                samples=[[],[]];orders=[]
                for i in range(60):
                    order=(0,1) if i%2==0 else (1,0);orders.append(order)
                    for idx in order:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graphs[idx].replay();end.record();end.synchronize()
                        samples[idx].append(start.elapsed_time(end)*1000/50)
                row['timing']={'calls_per_graph':50,'order':orders,'samples_us':samples,
                               'median_us':[statistics.median(s) for s in samples]}
                del graphs
            if a.mode=='ncu':
                fn=baseline if a.arm=='baseline' else candidate
                for _ in range(10):call(fn,dest[0])
                torch.cuda.synchronize()
                torch.cuda.profiler.start();call(fn,dest[0]);torch.cuda.synchronize();torch.cuda.profiler.stop()
                row['profile_arm']=a.arm
            row['status']='PASS'
            save(a.out/'results.json',report)
            print(json.dumps({key:row[key] for key in ('tokens','hidden','topk','capacity','status')}),flush=True)
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e))
        raise
    finally:
        report['finished_at']=time.time()
        save(a.out/'results.json',report)


if __name__=='__main__':main()
