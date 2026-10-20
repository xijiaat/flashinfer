set -uo pipefail
source /test/env.sh
cd /test/source
set +e
for variant in default aa; do
  extra=()
  if [ "$variant" = aa ]; then extra+=(--control); fi
  timeout 1800 python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 /test/scripts/compare_live_moe_default.py --candidate /test/scripts/hopper_mxfp4_final.py --out /test/raw/moe-v41-$variant --geometry v41 --experts 384 --default-tactic "${extra[@]}" > /test/raw/moe-v41-$variant.log 2>&1
  rc=$?
  printf '%s\n' "$rc" > /test/raw/moe-v41-$variant.exit
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
