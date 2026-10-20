set -uo pipefail
source /test/env.sh
cd /test/source
set +e
timeout 1800 python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 /test/scripts/compare_live_moe.py --candidate /test/scripts/hopper_mxfp4_candidate.py --out /test/raw/moe-v41 --geometry v41 --experts 384 > /test/raw/moe-v41.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/moe-v41.exit
exit "$rc"
