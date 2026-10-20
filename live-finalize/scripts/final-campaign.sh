set -uo pipefail
source /test/env.sh
bash /test/scripts/run_extra.sh
rc=$?
printf '%s\n' "$rc" > /test/raw/extra-campaign.exit
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
cd /test/candidate-source
export PYTHONPATH=/test/candidate-source:/prior/deps:/test/scripts
for test in official oracle; do
  if [ "$test" = official ]; then
    timeout 2400 bash tests/moe_ep/run_tests.sh mega_sm90_mxfp4 > /test/raw/final-official.log 2>&1
  else
    timeout 600 python -m pytest -q tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py > /test/raw/final-oracle.log 2>&1
  fi
  rc=$?
  printf '%s\n' "$rc" > /test/raw/final-$test.exit
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
printf '0\n' > /test/raw/final-campaign.exit
