#!/usr/bin/env python3
"""Summarize saved held-out episodes without GPU work or timing."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    rows = {}
    seen = set()
    expected = {(a,b) for a in manifest['protocol']['arms'] for b in manifest['protocol']['batch_sizes']}
    for path in sorted(directory.glob('*-B*-*.json')):
        d = json.loads(path.read_text())
        key = (d['arm'],d['batch'])
        episode = (*key,d['scenario'])
        if key not in expected or episode in seen or not 0 <= d['scenario'] < manifest['protocol']['scenarios']:
            raise ValueError(f'Unexpected/duplicate episode: {path}')
        if len(d['goal_outcomes']) != 5:
            raise ValueError(f'Incomplete goal outcomes: {path}')
        seen.add(episode)
        rows.setdefault(key,[]).append(d)
    summary = dict(source=manifest['source'], protocol_sha256=manifest['protocol_sha256'],
                   expected_scenarios=manifest['protocol']['scenarios'], rows=[])
    for (arm,batch), data in sorted(rows.items()):
        success = sum(all(g == 'reached' for g in d['goal_outcomes']) for d in data)
        feasible = lambda d: (d['telemetry']['applied_effort_ratio'] <= 1+1e-6 and
                             d['telemetry']['velocity_ratio'] <= 1+1e-6 and
                             d['telemetry']['position_violation_rad'] <= 1e-6)
        row = dict(arm=arm,batch=batch,episodes=len(data),successes=success,
                   feasible_episodes=sum(feasible(d) for d in data),
                   successful_and_feasible=sum(feasible(d) and all(g=='reached' for g in d['goal_outcomes']) for d in data),
                   maxima={k:max(d['telemetry'][k] for d in data) for k in
                       ('command_effort_ratio','applied_effort_ratio','velocity_ratio',
                        'position_violation_rad','payload_swing_max_rad','payload_relative_speed_max_rad_s')})
        summary['rows'].append(row)
    summary['complete'] = set(rows) == expected and all(len(v)==summary['expected_scenarios'] for v in rows.values())
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory),indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f: f.write(text)
    print(text)
