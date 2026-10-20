#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=0
ncu=/opt/nvidia/nsight-compute/2026.3.1/ncu
for m in 16 16384; do
  for arm in baseline candidate; do
    name=ncu-${m}-${arm}
    "$ncu" --profile-from-start off --clock-control none --cache-control none --launch-count 20 --kernel-name-base demangled --rename-kernels off --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__block_size,launch__grid_size --export /test/raw/$name /opt/block32/.venv/bin/python /test/scripts/compare_staging.py --candidate /test/candidate --out /test/raw/$name-results --mode ncu --arm "$arm" --tokens "$m" --hidden 7168 > /test/raw/$name.log 2>&1
    report=$(find /test/raw -maxdepth 1 -name "$name.ncu-rep*")
    "$ncu" --import "$report" --csv --page raw > /test/raw/$name.csv
  done
done
