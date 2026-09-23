#!/usr/bin/env python3
"""Validate retained count totals and regroup into 20 bins without model fitting."""
import csv
import json
from pathlib import Path
p=Path('results/mdpi_r2_review')
a=json.loads((p/'histogram_seed42.json').read_text())
rows=[]
report={}
for key,total in a['expected_totals'].items():
    counts=[v for group in a[key] for v in group]
    assert len(counts)==100 and sum(counts)==total,(key,len(counts),sum(counts),total)
    merged=[sum(counts[i:i+5]) for i in range(0,100,5)]
    report[key]={'n':total,'counts_20_bins':merged,'probabilities':[v/total for v in merged]}
    rows += [{'population':key,'bin_left':i/20,'bin_right':(i+1)/20,'count':v,'proportion':v/total} for i,v in enumerate(merged)]
out=p/'derived';out.mkdir(exist_ok=True)
with (out/'histogram_seed42_20bins.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['population','bin_left','bin_right','count','proportion'])
    writer.writeheader();writer.writerows(rows)
(out/'histogram_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('HISTOGRAM_TOTALS_VALIDATED')
print(json.dumps(report,indent=2))
