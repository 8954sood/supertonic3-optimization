"""Remeasure frozen engine versions in balanced, guarded fresh processes.

All variants share the current Python runtime and F1200 learned weights. A
separate v5-package row measures batch-one folding as well as the v5 kernel.
This is a benchmark of versioned native engines, not historical Python stacks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def williams_order(count, row):
    """Even-size Williams design: balance positions and preceding treatments."""
    first = [0]
    for offset in range(1, count):
        first.append((offset + 1) // 2 if offset % 2 else count - offset // 2)
    return [(value + row) % count for value in first]


def summarize(report):
    rows = []
    for group in report['groups']:
        for speed in group['speeds']:
            for case in report['cases']:
                grouped = {}
                for name in group['variants']:
                    jobs = [job for job in report['runs'] if job['group'] == group['name']
                            and job['speed'] == speed and job['name'] == name
                            and job['returncode'] == 0 and job['benchmark'] and job['benchmark']['complete']]
                    grouped[name] = {job['block']: next(row for row in job['benchmark']['rows']
                                                      if row['case'] == case) for job in jobs}
                reference = grouped[group['variants'][0]]
                for name, completed in grouped.items():
                    if not completed:
                        continue
                    rtfs = [row['median_rtf'] for row in completed.values()]
                    paired = [100 * (1 - row['median_seconds'] / reference[block]['median_seconds'])
                              for block, row in completed.items() if block in reference]
                    stages = {}
                    for stage in next(iter(completed.values()))['runs'][0]['stages']:
                        stages[stage] = statistics.median(row['runs'][0]['stages'][stage]
                                                         for row in completed.values())
                    quartiles = statistics.quantiles(rtfs, n=4, method='inclusive') if len(rtfs) > 1 else [rtfs[0]] * 3
                    rows.append(dict(group=group['name'], name=name, speed=speed, case=case,
                        repetitions=len(completed), median_rtf=statistics.median(rtfs),
                        min_rtf=min(rtfs), max_rtf=max(rtfs), q1_rtf=quartiles[0], q3_rtf=quartiles[2],
                        median_seconds=statistics.median(row['median_seconds'] for row in completed.values()),
                        audio_seconds=next(iter(completed.values()))['samples'] / next(iter(completed.values()))['sample_rate'],
                        median_process_cpu_seconds=statistics.median(row['runs'][0]['process_cpu_seconds'] for row in completed.values()),
                        median_stage_seconds=stages,
                        median_paired_time_reduction_percent=statistics.median(paired) if paired else None,
                        paired_time_reductions_percent=paired))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=6)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    variants = config['variants']
    names = [variant['name'] for variant in variants]
    if platform.system() != 'Linux' or args.blocks != len(variants) or len(variants) % 2:
        parser.error('Use Linux and one even-sized complete Williams cycle')
    folder = args.output.parent / (args.output.stem + '_runs')
    if args.output.exists() or folder.exists() or len(set(names)) != len(names):
        parser.error('Choose new output paths and distinct variant names')
    os.nice(max(0, 19 - os.nice(0)))
    folder.mkdir(parents=True)
    by_name = {variant['name']: variant for variant in variants}
    report = dict(started_utc=now(), complete=False, blocks=args.blocks, variants=variants,
        groups=config['groups'], cases=['ko_short', 'en_short'], voice='M1', steps=8,
        threads=1, affinity=[0], seed=42, platform=platform.platform(),
        cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')),
        mem_total_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith('MemTotal:'))),
        runtime_sha256=sha256(ROOT / 'runtime.py'), harness_sha256=sha256(__file__),
        libraries={name: sha256(variant['library']) for name, variant in by_name.items()},
        note='One warm-up and one timed synthesis per language in each fresh process. '
             'Six main treatments use an even-size Williams order; speed order reverses by block. '
             'Historical 1.05x check alternates v2/v5-package AB/BA. No timed sample is trimmed. '
             'RTF is wall seconds / final audio seconds; loading, hashing and file I/O excluded. '
             'All variants use the same current Python runtime. Shared x86 cloud server, not ARM. '
             'RSS guard: process tree 480 MB, system reserve 80 MB, sampled every 50 ms; not cgroup enforcement.',
        runs=[], wave_checks=[], summary=[])

    def save():
        report['summary'] = summarize(report)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(args.output)

    expected = {}
    for block in range(args.blocks):
        for group in config['groups']:
            treatments = group['variants']
            order = [treatments[index] for index in williams_order(len(treatments), block)]
            speeds = group['speeds'] if block % 2 == 0 else list(reversed(group['speeds']))
            for speed in speeds:
                current_order = order if group['speeds'].index(speed) % 2 == 0 else list(reversed(order))
                for name in current_order:
                    variant = by_name[name]
                    stem = f'{group["name"]}_{block}_{speed:g}_{name}'
                    raw, guard = folder / (stem + '.json'), folder / (stem + '_guard.json')
                    command = [sys.executable, str(ROOT / 'memory_guard.py'),
                        '--max-rss-mb', '480', '--reserve-mb', '80', '--seconds', '240',
                        '--stats', str(guard), '--', sys.executable,
                        str(ROOT / 'experiments/benchmark_server_case.py'),
                        '--model', variant['model'], '--runtime-root', '.',
                        '--speed', str(speed), '--voice', 'M1', '--affinity', '0',
                        '--cases', *report['cases'], '--repeats', '1', '--output', str(raw),
                        '--exact-library', variant['library']]
                    print('START', stem, flush=True)
                    stamp, started = now(), time.monotonic()
                    cpu_before = next(line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu0 '))
                    with (folder / (stem + '.log')).open('w') as log:
                        child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    benchmark = json.loads(raw.read_text()) if raw.exists() else None
                    job = dict(name=name, group=group['name'], block=block, speed=speed,
                        started_utc=stamp, elapsed_process_seconds=time.monotonic() - started,
                        returncode=child.returncode, benchmark=benchmark,
                        guard=json.loads(guard.read_text()) if guard.exists() else None,
                        cpu0_before=cpu_before,
                        cpu0_after=next(line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu0 ')))
                    report['runs'].append(job)
                    save()
                    if child.returncode or not benchmark or not benchmark['complete']:
                        raise RuntimeError(f'Guarded process failed: {stem}; original result preserved')
                    for row in benchmark['rows']:
                        key = (speed, row['case'])
                        actual = (row['float32_sha256'], row['samples'], row['sample_rate'], row['text'])
                        expected.setdefault(key, actual)
                        identical = expected[key] == actual
                        report['wave_checks'].append(dict(name=name, block=block, speed=speed,
                            case=row['case'], float32_bitidentical=identical))
                        if not identical:
                            save()
                            raise RuntimeError(f'Waveform mismatch: {stem} {row["case"]}')
                    save()
                    print('DONE', stem, [(row['case'], round(row['median_rtf'], 5))
                          for row in benchmark['rows']], flush=True)
    report['complete'] = True
    report['finished_utc'] = now()
    save()
    print('COMPLETE', len(report['runs']), 'fresh processes', flush=True)


if __name__ == '__main__':
    main()
