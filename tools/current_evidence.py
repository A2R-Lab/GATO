#!/usr/bin/env python3
"""Build compact evidence from trusted local GATO artifacts, or re-render its plot.

Extract: python tools/current_evidence.py --repo /path/to/GATO
Render bundled JSON only: python tools/current_evidence.py
Requires matplotlib for rendering, numpy to read the trusted local pickles.
Never pass an untrusted repository/pickle source. No GPU work or timing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'static/demos/current'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(repo):
    evidence = {'qualification': 'October 2026 exploratory results; not held-out, not hardware validation',
                'indy7': {}, 'fig7': {}}
    for backend in ('gato', 'bt'):
        path = repo / f'examples/benchmarks/data/sweep_fig8_{backend}_indy7.csv'
        cells = {}
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if int(row['N']) == 64:
                    cells[row['B']] = float(row['median_ms'])  # last saved row per cell
        evidence['indy7'][backend] = {'path': str(path.relative_to(repo)), 'sha256': digest(path),
                                      'N': 64, 'boundary': 'internal solver time', 'median_ms': cells}
    for tag in ('fig7_stop', 'fig7_pass_through', 'fig7_pickplace_v2_fe'):
        path = repo / f'examples/paper-figures/data/{tag}.pkl'
        with path.open('rb') as stream:
            data = pickle.load(stream)  # trusted local research artifact only
        evidence['fig7'][tag] = {'path': str(path.relative_to(repo)), 'sha256': digest(path),
            'source': data['source'], 'solver_params': data['solver_params'],
            'mpc_defaults': data['mpc_defaults'], 'simulation_protocol': data['simulation_protocol'],
            'successes': {str(b): [t is not None for t in data['pool'][b]] for b in data['batch_sizes']}}
    return evidence


def render(data, portrait=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'fe': '#2778c4', 'wid': '#d07616'}
    labels = {'fe': 'Exploration (FE)', 'wid': 'Identified weight + exploration'}
    fig, axes = plt.subplots(2 if portrait else 1, 1 if portrait else 2,
                             figsize=(5,7.3) if portrait else (10,4.1), sharey=True, layout='constrained')
    for ax, title, series in zip(axes, ('Arm arrival with dwell', 'Pass-through arm arrival'),
            ((('fe','fig7_stop'),), (('fe','fig7_pickplace_v2_fe'),('wid','fig7_pass_through')))):
        for method, tag in series:
            rows = data['fig7'][tag]['successes']
            batches = sorted(map(int,rows))
            rates = [100 * sum(rows[str(b)])/len(rows[str(b)]) for b in batches]
            ax.plot(batches,rates,marker='o' if method=='fe' else 's',color=colors[method],label=labels[method])
        ax.set(title=title,xlabel='Batch size',xscale='log',ylim=(0,105))
        ax.set_xticks([1,8,32,128],['1','8','32','128'])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    axes[0].set_ylabel('Episodes reaching all five goals (%)')
    fig.suptitle('Exploratory seed-0 pool · 100 scenarios' + ('\n' if portrait else ' · ') + 'no actuator limits',fontsize=11)
    fig.savefig(OUT/('fig7_exploratory_mobile.png' if portrait else 'fig7_exploratory.png'),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    args = parser.parse_args()
    path = OUT/'evidence_2026-10-04.json'
    if args.repo:
        data = extract(args.repo.resolve())
        path.write_text(json.dumps(data,indent=2)+'\n')
    else:
        data = json.loads(path.read_text())
    render(data)
    render(data,portrait=True)
