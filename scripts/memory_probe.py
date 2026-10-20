from pathlib import Path
import json,torch
from compare_staging import load_stagers,buffers,save
baseline,candidate=load_stagers(Path('/test/candidate'))
result=[]
for t,h,k in [(16,7168,8),(16384,7168,8)]:
 x=torch.randn((t,h),device='cuda',dtype=torch.bfloat16)
 ids=torch.zeros((t,k),device='cuda',dtype=torch.int64)
 weights=torch.ones((t,k),device='cuda')
 outputs=buffers(torch,t,h,k,t+32)
 for name,fn in [('baseline',baseline),('candidate',candidate)]:
  fn(x,weights,ids,*outputs,quantize_input=True,safe_quantization=True)
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()
  before=torch.cuda.memory_allocated()
  fn(x,weights,ids,*outputs,quantize_input=True,safe_quantization=True)
  torch.cuda.synchronize()
  result.append(dict(tokens=t,hidden=h,arm=name,temporary_peak_bytes=torch.cuda.max_memory_allocated()-before,live_delta_bytes=torch.cuda.memory_allocated()-before))
print(json.dumps(result))
save(Path('/test/raw/memory.json'),result)
