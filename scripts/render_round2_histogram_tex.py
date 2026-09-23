#!/usr/bin/env python3
"""Validate retained seed-42 histogram counts and emit self-contained PGFPlots.

No model fitting, interpolation of missing samples, or source data mutation.
Each population is normalized separately to empirical bin probability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('round2_histograms.tex'))
    args=ap.parse_args()
    source=ROOT/'results/mdpi_r2/review_evidence/seed42_histogram_counts.json'
    data=json.loads(source.read_text())
    parts=[r'\subsection{Known-versus-unknown score distributions}',
       'Figures~\\ref{fig:histSlowrateDoS} and \\ref{fig:histUDPFlood} show the retained 100-bin empirical score distributions for seed 42. Each population is normalized separately. They are illustrative single-seed diagnostics, not five-seed uncertainty summaries. All five-seed results are reported in the preceding tables.']
    report={}
    for family in ['SlowrateDoS','UDPFlood']:
        rec=data[family]
        for population in ['known','unknown']:
            counts=rec[population]
            assert len(counts)==100, (family,population,len(counts))
            assert all(isinstance(x,int) and x>=0 for x in counts)
            assert sum(counts)==rec['n_'+population], (family,population,sum(counts),rec['n_'+population])
        report[family]={'known_count':sum(rec['known']),'unknown_count':sum(rec['unknown']),'bins':100}
        name='histdata'+family
        parts.append(r'\pgfplotstableread[col sep=comma]{')
        parts.append('score,known,unknown')
        for i,(known,unknown) in enumerate(zip(rec['known'],rec['unknown'])):
            parts.append(f'{(i+.5)/100:.3f},{known/rec["n_known"]:.12g},{unknown/rec["n_unknown"]:.12g}')
        parts.extend(['}\\'+name,
            r'\begin{figure}[htbp]\centering',
            r'\begin{tikzpicture}\begin{axis}[width=.9\linewidth,height=58mm,xmin=0,xmax=1,ymin=0,xlabel={Decoded composite score},ylabel={Empirical probability per 0.01 bin},legend style={at={(.98,.98)},anchor=north east}]',
            r'\addplot+[const plot,mark=none] table[x=score,y=known]{\'+name+'};',
            r'\addplot+[const plot,mark=none] table[x=score,y=unknown]{\'+name+'};',
            r'\legend{Known traffic,Held-out attack}',
            r'\end{axis}\end{tikzpicture}',
            '\\caption{'+family+': corrected 16Q score histogram, seed 42. Known $n='+str(rec['n_known'])+'$; withheld $n='+str(rec['n_unknown'])+'$. Each population\'s bin masses sum to one; the last bin includes score one. This plot is not used to infer an unreported calibration threshold.}\\label{fig:hist'+family+'}',
            r'\end{figure}'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    print(json.dumps({'histogram_validation':'passed','populations':report},indent=2))

if __name__=='__main__':main()
