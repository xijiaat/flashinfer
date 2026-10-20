"""Retain all frozen A/B cases; no filtering by observed benefit."""
from pathlib import Path
import csv
import json
import statistics

run=Path(__file__).resolve().parents[1]
raw=run/'raw'
rows=[]
def pct(v,p):
 s=sorted(v);i=(len(s)-1)*p;lo=int(i);hi=min(lo+1,len(s)-1)
 return s[lo]+(s[hi]-s[lo])*(i-lo)
for name in ('small','large','v41','control'):
 reports=[json.loads((raw/('moe-'+name)/('rank%d.json'%rank)).read_text()) for rank in range(4)]
 for report in reports:
  assert report['status']=='PASS' and report['cleanup_phase']=='complete'
  assert len(report['cases'])==8
  if name!='control':assert report['candidate_dispatch_cache']['misses']>0
 for index,case in enumerate(reports[0]['cases']):
  assert case['status']=='PASS' and len(case['rounds'])==3
  for other in reports[1:]:
   assert other['cases'][index]['rounds']==case['rounds']
   assert other['cases'][index]['status']=='PASS'
  a=[v for rr in case['rounds'] for v in rr['samples_us'][0]]
  b=[v for rr in case['rounds'] for v in rr['samples_us'][1]]
  assert len(a)==len(b)==60
  meda,medb=statistics.median(a),statistics.median(b)
  row=dict(geometry=name,H=reports[0]['hidden'],I=reports[0]['intermediate'],E=reports[0]['experts'],K=reports[0]['topk'],M=case['tokens_per_rank_nominal'],routing=case['routing'],baseline_us=meda,candidate_us=medb,speedup_pct=100*(meda/medb-1),latency_reduction_pct=100*(1-medb/meda),baseline_p10=pct(a,.1),baseline_p90=pct(a,.9),candidate_p10=pct(b,.1),candidate_p90=pct(b,.9))
  for i,rr in enumerate(case['rounds']):row['round%d_speedup_pct'%(i+1)]=100*(rr['median_us'][0]/rr['median_us'][1]-1)
  rows.append(row)
(run/'tables/complete-moe-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
with (run/'tables/complete-moe-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:
 print('%-7s M%-4d %-10s %.3f -> %.3f us  %+6.2f%%  rounds %+.2f/%+.2f/%+.2f'%(r['geometry'],r['M'],r['routing'],r['baseline_us'],r['candidate_us'],r['speedup_pct'],r['round1_speedup_pct'],r['round2_speedup_pct'],r['round3_speedup_pct']))
