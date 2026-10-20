"""Recompute all summaries from retained full rank-max samples; no filtering."""
import csv
import io
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = []
for cohort in ('moe-v41', 'moe-v41-default', 'moe-v41-aa'):
    p = root / 'raw' / cohort / 'rank0.json'
    if not p.exists():
        continue
    ranks = [json.loads((p.parent / f'rank{rank}.json').read_text()) for rank in range(4)]
    assert all(d['status'] == 'PASS' and d['cleanup_phase'] == 'complete' for d in ranks)
    for case in ranks[0]['cases']:
        assert case['status'] == 'PASS' and len(case['rounds']) == 3
        merged = [[], []]
        gains = []
        for rd in case['rounds']:
            assert all(len(samples) == 20 for samples in rd['samples_us'])
            for arm in (0, 1):
                assert rd['samples_us'][arm] == [max(x) for x in rd['all_rank_us'][arm]]
                merged[arm].extend(rd['samples_us'][arm])
            a, b = rd['median_us']
            gains.append((a / b - 1) * 100)
        a, b = map(statistics.median, merged)
        rows.append(dict(cohort=cohort, tokens=case['tokens_per_rank_nominal'], routing=case['routing'],
                         baseline_us=a, candidate_us=b, latency_reduction_pct=(1-b/a)*100,
                         speedup_pct=(a/b-1)*100, round_speedups_pct=gains,
                         sample_min_us=list(map(min, merged)), sample_max_us=list(map(max, merged)),
                         stdev_us=list(map(statistics.stdev, merged)), samples_per_arm=60,
                         rounds=case['rounds'], checks=case['checks'],
                         tactic=case.get('effective_tactic'), memory=case.get('arm_warmup_memory')))
(root/'tables').mkdir(exist_ok=True)
(root/'tables/complete-summary.json').write_text(json.dumps(rows, indent=2))
keys = ['cohort', 'tokens', 'routing', 'baseline_us', 'candidate_us', 'latency_reduction_pct', 'speedup_pct', 'round_speedups_pct', 'sample_min_us', 'sample_max_us', 'stdev_us', 'samples_per_arm']
with (root/'tables/complete-summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows({k:r[k] for k in keys} for r in rows)
ncu = {}
for arm in ('baseline', 'live'):
    data = list(csv.reader(io.StringIO((root/'raw'/f'reduce-{arm}.csv').read_text())))
    i = next(i for i, row in enumerate(data) if row and row[0] == 'ID')
    metrics = {k: {'value': v, 'unit': u} for k,u,v in zip(data[i], data[i+1], data[i+2]) if any(s in k for s in ('gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum', 'launch__grid_size'))}
    ncu[arm] = metrics
(root/'tables/ncu-summary.json').write_text(json.dumps(ncu, indent=2))
for row in rows:
    print(f"{row['cohort']} M{row['tokens']} {row['routing']}: {row['baseline_us']:.3f} -> {row['candidate_us']:.3f} us, {row['speedup_pct']:+.3f}%")
