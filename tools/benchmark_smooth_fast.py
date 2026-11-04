"""Serial, interleaved prepared-input joint-fast/default reconstruction benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runtime/validation/default_smooth_fast'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    dest = WORK / a.batch
    dest.mkdir(parents=True, exist_ok=False)
    snapshot = WORK / 'source_before'
    seq = ROOT / 'runtime/outputs/zhoupei_cap/prepared/sequence.json'
    records = []
    for i in range(a.rounds):
        for label, cwd, flags in (
            ('joint_fast', snapshot, ['--shape-fusion', 'joint', '--fast', '--mid-rounds', '1',
             '--chain', 'full', '--postfit', 'full', '--steps', '240',
             '--support-min-information', '0', '--coverage-allowance-ratio', '0',
             '--anchor-prior-weight', '0']),
            ('default', ROOT, []),
        ):
            out = dest / f'{label}_{i+1}'
            cmd = [sys.executable, '-m', 'head_rgbd', '--runtime', str(ROOT / 'runtime'),
                   'reconstruct', '--sequence', str(seq), '--output', str(out), *flags]
            env = dict(os.environ, PYTHONPATH=str(cwd), PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            print('START', label, i+1, json.dumps(cmd), flush=True)
            start = time.perf_counter()
            with (dest / f'{label}_{i+1}.log').open('x', encoding='utf-8') as log:
                r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            wall = time.perf_counter() - start
            report = json.loads((out / 'run_report.json').read_text()) if (out / 'run_report.json').exists() else {}
            records.append(dict(label=label, repetition=i+1, command=cmd, cwd=str(cwd),
                output=str(out), exit_code=r.returncode, wall_s=wall, run_report_s=report.get('elapsed_s'),
                stages=[dict(name=s['name'], elapsed_s=s['elapsed_s'], exit_code=s['exit_code'])
                        for s in report.get('stages', [])]))
            payload = dict(sequence=str(seq), sequence_sha256=hashlib.sha256(seq.read_bytes()).hexdigest(),
                timing='external subprocess wall clock: imports, asset gate, environment, reconstruction, export; '
                       'prepared reused identically; capture/prepare/evaluation excluded',
                baseline_source_manifest=str(WORK / 'source_before_sha256.json'), records=records)
            for name in ('joint_fast', 'default'):
                times = [r['wall_s'] for r in records if r['label'] == name]
                payload[name] = dict(all_s=times, median_s=statistics.median(times) if times else None,
                                     min_s=min(times) if times else None, max_s=max(times) if times else None)
            (dest / 'timings.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print('END', label, i+1, 'exit', r.returncode, 'wall_s', wall, flush=True)
            if r.returncode:
                print((dest / f'{label}_{i+1}.log').read_text()[-5000:], flush=True)
                raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
