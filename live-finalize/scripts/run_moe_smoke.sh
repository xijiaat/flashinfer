#!/bin/bash
set -uo pipefail
source /test/env.sh
cd /test/source
set +e
timeout 900 python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 /test/scripts/compare_live_moe.py --candidate /test/scripts/hopper_mxfp4_candidate.py --out /test/raw/moe-smoke --geometry tiny --experts 16 --smoke > /test/raw/moe-smoke.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/moe-smoke.exit
exit "$rc"
