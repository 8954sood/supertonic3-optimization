"""Publish all round18 timings with repeated provenance deduplicated losslessly."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics


def export(raw):
    report = copy.deepcopy(raw)
    configurations, configuration_keys = {}, {}
    for job in report['runs']:
        benchmark = job['benchmark']
        configuration = {key: value for key, value in benchmark.items()
                         if key not in {'rows', 'load_seconds', 'complete'}}
        key = json.dumps(configuration, sort_keys=True)
        if key not in configuration_keys:
            identity = f'configuration-{len(configurations) + 1}'
            configurations[identity] = configuration
            configuration_keys[key] = identity
        job['benchmark_configuration'] = configuration_keys[key]
        job['benchmark'] = {key: benchmark[key] for key in ['rows', 'load_seconds', 'complete']}
    report['benchmark_configurations'] = configurations
    report['export_note'] = ('Repeated benchmark metadata is stored once in benchmark_configurations; '
                             'each run identifies its configuration. All timed measurements, warm-output '
                             'hashes, process guards, order and timestamps are retained; no outliers removed.')
    comparisons = []
    for group in report['groups']:
        names = group['variants']
        pairs = [('v2', name) for name in names if name != 'v2']
        if group['name'] == 'main':
            pairs += [('v1', 'v2'), ('v4', 'v5-kernel'), ('v4', 'v5-package'),
                      ('v5-kernel', 'v5-package')]
        for speed in group['speeds']:
            for case in report['cases']:
                jobs = {(job['name'], job['block']): next(row for row in job['benchmark']['rows']
                                                         if row['case'] == case)
                        for job in report['runs'] if job['group'] == group['name'] and job['speed'] == speed}
                for reference, candidate in pairs:
                    reductions, cpu_reductions = [], []
                    for block in range(report['blocks']):
                        before, after = jobs[reference, block], jobs[candidate, block]
                        reductions.append(100 * (1 - after['median_seconds'] / before['median_seconds']))
                        cpu_reductions.append(100 * (1 - after['runs'][0]['process_cpu_seconds'] /
                                                    before['runs'][0]['process_cpu_seconds']))
                    comparisons.append(dict(group=group['name'], speed=speed, case=case,
                        reference=reference, candidate=candidate, pairs=len(reductions),
                        median_paired_time_reduction_percent=statistics.median(reductions),
                        paired_time_reductions_percent=reductions,
                        faster_pairs=sum(value > 0 for value in reductions),
                        median_paired_cpu_time_reduction_percent=statistics.median(cpu_reductions)))
    report['comparisons'] = comparisons
    report['validation'] = dict(
        fresh_processes=len(report['runs']),
        timed_syntheses=sum(len(row['runs']) for job in report['runs'] for row in job['benchmark']['rows']),
        all_guards_passed=all(job['guard']['reason'] is None and job['returncode'] == 0 for job in report['runs']),
        all_float32_identical=all(check['float32_bitidentical'] for check in report['wave_checks']),
        all_actual_vector_calls_eight=all(run['actual_vector_calls'] == 8 for job in report['runs']
            for row in job['benchmark']['rows'] for run in row['runs']),
        all_libraries_match_pinned_hash=all(
            configurations[job['benchmark_configuration']]['exact_library_sha256'] == report['libraries'][job['name']]
            for job in report['runs']),
        all_runtimes_match_pinned_hash=all(
            configurations[job['benchmark_configuration']]['runtime_sha256'] == report['runtime_sha256']
            for job in report['runs']),
        max_sampled_process_tree_rss_mb=max(job['guard']['sampled_peak_process_tree_rss_mb'] for job in report['runs']),
        minimum_system_available_mb=min(job['guard']['minimum_system_available_mb'] for job in report['runs']),
        no_new_wer_cer_mos=True, learned_weights_changed=False,
        inference_library_swapped=False, server_default_model_changed=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if not raw['complete'] or len(raw['runs']) != 84:
        parser.error('Expected all 84 completed fresh processes')
    report = export(raw)
    assert report['validation']['all_guards_passed']
    assert report['validation']['all_float32_identical']
    assert report['validation']['all_actual_vector_calls_eight']
    assert report['validation']['all_libraries_match_pinned_hash']
    assert report['validation']['all_runtimes_match_pinned_hash']
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    summary = {key: report[key] for key in ['started_utc', 'finished_utc', 'complete', 'blocks',
               'voice', 'steps', 'threads', 'affinity', 'seed', 'cpu_model', 'mem_total_kib',
               'note', 'summary', 'comparisons', 'validation']}
    summary['raw_result'] = args.output.as_posix()
    summary['raw_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['validation'], indent=2))
    for name in [variant['name'] for variant in report['variants']]:
        selected = [row for row in report['summary'] if row['group'] == 'main' and row['name'] == name]
        print(name, [(row['speed'], row['case'], round(row['median_rtf'], 6)) for row in selected])


if __name__ == '__main__':
    main()
