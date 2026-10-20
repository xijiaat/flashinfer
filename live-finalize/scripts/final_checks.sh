set -uo pipefail
source /test/env.sh
cd /test/candidate-source
export PYTHONPATH=/test/candidate-source:/prior/deps:/test/scripts
timeout 600 python -m pytest -q tests/moe_ep/test_sm90_mxfp4_fused_staging.py > /test/raw/final-staging.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/final-staging.exit
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
cd /test/source
export PYTHONPATH=/test/source:/prior/deps:/test/scripts
timeout 600 python /test/scripts/negative_control.py > /test/raw/negative-control.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/negative-control.exit
# Expected failure must be classified from the exact assertion, not this exit alone.
exit 0
