"""Alternate frozen engines in fresh guarded processes without swapping libraries."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='JSON list: name, model, optional library and runtime_root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--speeds', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--voices', nargs='+', default=['M1'])
    parser.add_argument('--cases', nargs='+', default=['ko_short', 'en_short'])
    parser.add_argument('--max-rss-mb', type=float, default=480)
    parser.add_argument('--reserve-mb', type=float, default=80)
    args = parser.parse_args()
    folder = args.output.parent / (args.output.stem + '_runs')
    if platform.system() != 'Linux' or args.pairs < 3:
        parser.error('Use Linux and at least three pairs')
    if args.output.exists() or folder.exists():
        parser.error('Choose a new output')
    variants = json.loads(args.config.read_text())
    if len(variants) < 2 or len({v['name'] for v in variants}) != len(variants):
        parser.error('Need two or more distinctly named variants')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', v['name']) for v in variants):
        parser.error('Variant names must be plain filename components')
    os.nice(max(0, 19 - os.nice(0)))
    folder.mkdir(parents=True)
    report = dict(variants=variants, pairs=args.pairs, steps=8, threads=1, affinity=[0],
        note='Fresh single-engine processes, rotated/reversed order, one warm-up and one timed '
             'run per text. Sampled RSS guard; no library swaps or concurrent inference.',
        runs=[], summary=[], wave_checks=[], complete=False)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    for voice in args.voices:
        for pair in range(args.pairs):
            offset = pair % len(variants)
            order = variants[offset:] + variants[:offset]
            for speed_index, speed in enumerate(args.speeds):
                for variant in (order if speed_index % 2 == 0 else list(reversed(order))):
                    name = f'{voice}_{speed:g}_{pair}_{variant["name"]}'
                    raw, guard = folder / (name + '.json'), folder / (name + '_guard.json')
                    command = [sys.executable, str(ROOT / 'memory_guard.py'),
                        '--max-rss-mb', str(args.max_rss_mb), '--reserve-mb', str(args.reserve_mb),
                        '--seconds', '240', '--stats', str(guard), '--', sys.executable,
                        str(ROOT / 'experiments/benchmark_server_case.py'),
                        '--model', variant['model'], '--runtime-root', variant.get('runtime_root', '.'),
                        '--speed', str(speed), '--voice', voice, '--affinity', '0',
                        '--cases', *args.cases, '--repeats', '1', '--output', str(raw)]
                    if variant.get('library'):
                        command += ['--exact-library', variant['library']]
                    print('START', name, flush=True)
                    with (folder / (name + '.log')).open('w') as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    benchmark = json.loads(raw.read_text()) if raw.exists() else None
                    stats = json.loads(guard.read_text()) if guard.exists() else None
                    report['runs'].append(dict(name=variant['name'], voice=voice, pair=pair,
                        speed=speed, returncode=result.returncode, benchmark=benchmark, guard=stats))
                    save()
                    if result.returncode or not benchmark or not benchmark['complete']:
                        raise RuntimeError(f'Guarded process failed: {name}')
                    batch = [r for r in report['runs'] if r['voice'] == voice and
                             r['speed'] == speed and r['pair'] == pair]
                    baseline_job = next((r for r in batch if r['name'] == variants[0]['name']), None)
                    if baseline_job:
                        expected = {r['case']: (r['float32_sha256'], r['samples'], r['sample_rate'], r['text'])
                                    for r in baseline_job['benchmark']['rows']}
                        for job in batch:
                            actual = {r['case']: (r['float32_sha256'], r['samples'], r['sample_rate'], r['text'])
                                      for r in job['benchmark']['rows']}
                            if actual != expected:
                                raise RuntimeError('Candidate waveform differs in the current pair')
                    print('DONE', name, [(r['case'], round(r['median_rtf'], 4))
                          for r in benchmark['rows']], flush=True)
    reference = variants[0]['name']
    for voice in args.voices:
        for speed in args.speeds:
            for case in args.cases:
                groups = {}
                for variant in variants:
                    jobs = [r for r in report['runs'] if r['name'] == variant['name'] and
                            r['voice'] == voice and r['speed'] == speed]
                    groups[variant['name']] = [next(c for c in j['benchmark']['rows']
                                                   if c['case'] == case) for j in jobs]
                baseline = groups[reference]
                for name, rows in groups.items():
                    identical = all((b['float32_sha256'], b['samples'], b['sample_rate'], b['text']) ==
                                    (r['float32_sha256'], r['samples'], r['sample_rate'], r['text'])
                                    for b, r in zip(baseline, rows))
                    report['wave_checks'].append(dict(voice=voice,speed=speed,case=case,
                                                       name=name,float32_bitidentical=identical))
                    save()
                    if not identical:
                        raise RuntimeError('Candidate changed generated waveform')
                    reductions = [100 * (1 - r['median_seconds'] / b['median_seconds'])
                                  for b, r in zip(baseline, rows)]
                    report['summary'].append(dict(name=name,voice=voice,speed=speed,case=case,
                        median_rtf=statistics.median(r['median_rtf'] for r in rows),
                        median_seconds=statistics.median(r['median_seconds'] for r in rows),
                        paired_time_reductions_percent=reductions,
                        median_paired_time_reduction_percent=statistics.median(reductions)))
    report['complete'] = True
    save()
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
