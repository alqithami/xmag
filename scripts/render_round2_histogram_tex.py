#!/usr/bin/env python3
"""Validate retained seed-42 counts and emit self-contained PGFPlots.

This utility does not fit models or replace measurements. Each population is
normalized separately to empirical bin probability. Output is plotting code,
not publication source committed to the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    parts = [
        r'\subsection{Known-versus-unknown score distributions}',
        'Figures~\\ref{fig:histSlowrateDoS} and \\ref{fig:histUDPFlood} show '
        'the retained 100-bin empirical score distributions for seed 42. '
        'Each population is normalized separately. These single-seed plots '
        'are illustrative diagnostics, not five-seed uncertainty summaries.',
    ]
    report = {}
    for family in ['SlowrateDoS', 'UDPFlood']:
        rec = data[family]
        for population in ['known', 'unknown']:
            counts = rec[population]
            expected = rec['n_' + population]
            if len(counts) != 100 or not all(type(x) is int and x >= 0 for x in counts):
                raise ValueError(f'Invalid histogram counts: {family}/{population}')
            if expected <= 0 or sum(counts) != expected:
                raise ValueError(f'Population count mismatch: {family}/{population}')
        report[family] = {
            'known_count': sum(rec['known']),
            'unknown_count': sum(rec['unknown']),
            'bins': 100,
        }
        name = 'histdata' + family
        macro = '\\' + name
        parts.extend([r'\pgfplotstableread[col sep=comma]{', 'score,known,unknown'])
        for i, (known, unknown) in enumerate(zip(rec['known'], rec['unknown'])):
            parts.append(f'{(i + 0.5) / 100:.3f},{known / rec["n_known"]:.12g},{unknown / rec["n_unknown"]:.12g}')
        parts.extend([
            '}' + macro,
            r'\begin{figure}[htbp]\centering',
            r'\begin{tikzpicture}\begin{axis}[width=.9\linewidth,height=58mm,xmin=0,xmax=1,ymin=0,xlabel={Decoded composite score},ylabel={Empirical probability per 0.01 bin},legend style={at={(.98,.98)},anchor=north east}]',
            r'\addplot+[const plot,mark=none] table[x=score,y=known]{' + macro + '};',
            r'\addplot+[const plot,mark=none] table[x=score,y=unknown]{' + macro + '};',
            r'\legend{Known traffic,Held-out attack}',
            r'\end{axis}\end{tikzpicture}',
            '\\caption{' + family + ': corrected 16Q score histogram, seed 42. '
            'Known $n=' + str(rec['n_known']) + '$; withheld $n=' + str(rec['n_unknown']) + '$. '
            'Each population\'s bin masses sum to one; the last bin includes score one. '
            'This plot does not infer an unreported calibration threshold.}'
            '\\label{fig:hist' + family + '}',
            r'\end{figure}',
        ])
    return '\n'.join(parts) + '\n', report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('round2_histograms.tex'))
    args = parser.parse_args()
    source = ROOT / 'results/mdpi_r2/review_evidence/seed42_histogram_counts.json'
    latex, report = render(json.loads(source.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding='utf-8')
    print(json.dumps({'histogram_validation': 'passed', 'populations': report}, indent=2))


if __name__ == '__main__':
    main()
