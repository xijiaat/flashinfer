import sys, importlib, os, json
from pathlib import Path
from compare_staging import load_stagers,PKG
base,candidate=load_stagers(Path('/test/candidate'))
for suffix in ('.staging','.backend','.tuner'):
    mod=importlib.import_module(PKG+suffix)
    if hasattr(mod,'stage_mega_moe_inputs'):
        assert mod.stage_mega_moe_inputs is base
        mod.stage_mega_moe_inputs=candidate
os.chdir('/test/source')
import pytest
rc=pytest.main(sys.argv[1:])
safe=sys.modules[PKG+'._fusion_candidate_safe']
cache=safe._compiled_bf16_stager.cache_info()
Path('/test/raw/candidate-suite-dispatch-rank'+os.environ.get('LOCAL_RANK','unit')+'.json').write_text(json.dumps({'exit':rc,'cache':cache._asdict()}))
if rc==0: assert cache.misses>0,'candidate BF16 kernel never compiled'
raise SystemExit(rc)
