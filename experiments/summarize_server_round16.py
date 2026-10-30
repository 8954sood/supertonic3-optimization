"""Summarize fresh-process EPYC measurements without extrapolating ARM gains."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    raw = json.loads(args.input.read_text())
    assert raw['complete']
    passed = [r for r in raw['runs'] if r['returncode'] == 0]
    assert all(r['benchmark']['complete'] for r in passed)
    assert all(b['actual_vector_calls'] == 8 for r in passed
               for case in r['benchmark']['rows'] for b in case['runs'])
    report = dict(source_commit=raw['source_commit'], snapshot=raw['snapshot'],
                  scope='Actual shared EPYC 1GB; fresh processes alternate. '
                        'Warm synthesis excludes loading. Not an ARM measurement.',
                  rows=[], extra_rows=[], failed_runs=[], complete=True)
    for run in raw['runs']:
        if run['returncode']:
            report['failed_runs'].append(dict(label=run['label'], speed=run['speed'],
                                             pair=run['pair'], guard=run['guard']))
    primary = [r for r in passed if not r['extra']]
    for speed in sorted({r['speed'] for r in primary}):
        for case in ['ko_short', 'en_short']:
            groups = {}
            for label in ['old_v4', 'latest_v4', 'latest_portable']:
                values = [(r['pair'], next(c for c in r['benchmark']['rows'] if c['case'] == case))
                          for r in primary if r['label'] == label and r['speed'] == speed]
                if values:
                    groups[label] = dict(pairs=[pair for pair, _ in values],
                        seconds=[c['median_seconds'] for _, c in values],
                        rtfs=[c['median_rtf'] for _, c in values],
                        median_seconds=statistics.median(c['median_seconds'] for _, c in values),
                        median_rtf=statistics.median(c['median_rtf'] for _, c in values),
                        wave_hashes=sorted({c['float32_sha256'] for _, c in values}))
            reference = groups['old_v4']
            assert len(reference['pairs']) == raw['pairs']
            assert len(reference['wave_hashes']) == 1
            reference_times = dict(zip(reference['pairs'], reference['seconds']))
            for label, group in groups.items():
                group['complete_pairs'] = group['pairs'] == reference['pairs']
                if label == 'latest_v4':
                    assert group['complete_pairs']
                group['paired_time_reductions_percent_vs_old_v4'] = [
                    100 * (1 - candidate / reference_times[pair])
                    for pair, candidate in zip(group['pairs'], group['seconds'])]
                group['median_paired_time_reduction_percent_vs_old_v4'] = statistics.median(
                    group['paired_time_reductions_percent_vs_old_v4'])
                group['wave_bitidentical_to_old_v4'] = group['wave_hashes'] == reference['wave_hashes']
            if 'latest_portable' in groups:
                latest = dict(zip(groups['latest_v4']['pairs'], groups['latest_v4']['seconds']))
                portable = groups['latest_portable']
                portable['paired_time_reductions_percent_vs_latest_v4'] = [
                    100 * (1 - candidate / latest[pair])
                    for pair, candidate in zip(portable['pairs'], portable['seconds'])]
                portable['median_paired_time_reduction_percent_vs_latest_v4'] = statistics.median(
                    portable['paired_time_reductions_percent_vs_latest_v4'])
            report['rows'].append(dict(speed=speed, case=case, groups=groups))
    for run in passed:
        if run['extra']:
            report['extra_rows'].extend(dict(label=run['label'], speed=run['speed'], **case)
                                        for case in run['benchmark']['rows'])
    report['sampled_peak_process_tree_rss_mb'] = max(
        r['guard']['sampled_peak_process_tree_rss_mb'] for r in passed)
    report['minimum_system_available_mb'] = min(r['guard']['minimum_system_available_mb'] for r in passed)
    report['successful_processes'] = len(passed)
    report['all_actual_vector_calls_eight'] = True
    report['quality_scope'] = 'Wave hashes compare unchanged speech on this server. No new ASR/WER/MOS.'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
