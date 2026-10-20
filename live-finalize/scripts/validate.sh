set -uo pipefail
source /test/env.sh
set -e
cp -a /test/source /test/candidate-source
tar xzf /test/fi-live-final-code.tgz -C /test/candidate-source
chmod +x /test/torchrun
export PYTHONPATH=/test/candidate-source:/prior/deps:/test/scripts
cd /test/candidate-source
set +e
timeout 2400 bash tests/moe_ep/run_tests.sh mega_sm90_mxfp4 > /test/raw/official-candidate.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/official-candidate.exit
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
timeout 600 python -m pytest -q tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py > /test/raw/oracle-candidate.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/oracle-candidate.exit
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
timeout 600 python -m pytest -q tests/moe_ep/test_sm90_mxfp4_fused_staging.py > /test/raw/staging-candidate.log 2>&1
rc=$?
printf '%s\n' "$rc" > /test/raw/staging-candidate.exit
exit "$rc"
