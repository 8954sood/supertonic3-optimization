"""Rebuild the selected engine and cleanup state from archived measurements."""
import json
from pathlib import Path
import statistics


def read(name):
    return json.loads((Path('results')/(name+'.json')).read_text())


def main():
    paired = read('round9_paired_tile_speed2')
    final = read('round9_v4_finalize')
    cleanup = read('round9_weight_cleanup')
    assert paired['complete'] and paired['original_library_restored']
    assert final['complete'] and final['selected_active_library'] == 'v4'
    assert all(r['bitidentical'] for r in paired['wave_checks'])
    assert all(r['float32_bitidentity'] for r in final['wave_checks'])
    assert all(r['returncode'] == 0 and r['guard']['reason'] is None for r in final['runs'])
    measurements = []
    for summary in paired['summary']:
        case = summary['case']
        candidate = [next(v for v in r['benchmark']['rows'] if v['case'] == case)
            for r in paired['runs'] if r['label'] == 'candidate']
        measurements.append(dict(case=case, reference_seconds=summary['reference_seconds'],
            candidate_seconds=summary['candidate_seconds'],
            median_paired_speedup=statistics.median(summary['paired_speedups']),
            paired_speedups=summary['paired_speedups'],
            candidate_rtf=statistics.median(r['median_rtf'] for r in candidate)))
    state = dict(round=9, date='2026-10-29', status='selected enginev4; cleanup complete',
        selected_engine=4, baseline_engine=2, model='F1200, unchanged learned tensors and quantization',
        actual_steps=8, threads=1, fast_generation_speed=1.35, fast_output_speed=2.0,
        waveform_processing=False, engine_change='M4N16 tiles with v2 safe-K8 widening; M8N8 exact fallback for other shapes',
        target='Linux x86_64 AVX2, tested EPYC7551 one logical core /1GB; no ARM timing inference',
        fast_paired_measurements=measurements,
        waveform_identity_checks=len(paired['wave_checks'])+len(final['wave_checks']),
        exact_abi_cases=len(read('round9_v4_abi')['rows']),
        normal_warm_measurements=[{k:r[k] for k in ['case','median_seconds','median_rtf']}
            for r in read('round9_v4_normal_server')['rows']],
        long_fast_warm_measurements=[{k:r[k] for k in ['case','median_seconds','median_rtf']}
            for r in read('round9_v4_long_server')['rows']],
        final_peak_sampled_process_tree_rss_mb=max(r['guard']['sampled_peak_process_tree_rss_mb'] for r in final['runs']),
        active_library_sha256=final['active_library_sha256'],
        cleanup={k:cleanup[k] for k in ['deleted_files','logical_bytes_deleted','available_bytes_before','available_bytes_after','kept_model_directories','kept_checkpoints']},
        rejected=['K32/K64 safety grouping: mostly slower','Exact epilogue fusion: no consistent gain over tile alone'],
        limitations=['Shared-server timings vary; one normal/long batch is not a paired speed proof',
            'RTF about1.00 at1x and about1.32..1.35 at2x does not establish realtime',
            'Round8 expanded Korean pronunciation errors remain; no new ASR quality improvement is claimed',
            'Legacy deleted model folders can retain manifests without runnable weights'],
        records='records/2026-10-29/ROUND9_ENGINE.md',
        resume_priority='Keep eight steps and original50 teacher policy; optimize runtime without changing accepted waveform. See round8 for separate remaining pronunciation work.')
    path=Path('records/2026-10-29/round9_state.json')
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
