# 작업 기록 색인

- [최신: 배속·원본50 비교·2배속 실패 원인·v3 미채택](records/2026-10-29/ROUND7_SPEED.md)
- [7차 기계 판독 상태](records/2026-10-29/round7_state.json)
- [모델 고정·실제1코어 서버 엔진 개선](records/2026-10-29/ROUND6_ENGINE.md)
- [5차: 원본50 교사·품질 개선·청취 확인](records/2026-10-29/ROUND5_LOCAL.md)

- [4차: 추가 그래프 실험·확대1024 학습·독립 평가](records/2026-10-29/ROUND4.md)
- [최신 인계](records/2026-10-29/HANDOFF_ROUND4.md)
- [4차 수치 집계](records/2026-10-29/round4_metrics.json)
- [Hugging Face 업로드 완료·공개 다운로드·재현 확인](records/2026-10-29/PUBLICATION.md)

현재 사용자가 선택한 기준 모델은 `models/round5-original50-student`(F1200)입니다. 아래B 속도는 과거 기록입니다. 이전 청취 기준 후보는 `models/cfg-mixed-v1-selected`였습니다. CPU1·실제 8스텝, 원본 CFG4를 단일 경로로 증류한 근사입니다. EPYC 1GB에서 KO5.823초/EN5.458초, 표본 최대RSS394MB. 작은 1024 후보는 더 빠르지만 한국어 오류가 늘어 대체하지 않았습니다.

- [최신 인계와 확정 결과](records/2026-10-28/HANDOFF_ROUND3.md)
- [3차: CFG 증류·8회 음성 손실·보코더 축소 실패](records/2026-10-28/ROUND3.md)
- [3차 원시 측정 및 학습 집계](records/2026-10-28/round3_metrics.json)
- [2차: 음질 원인 분리·혼합 폭 증류](records/2026-10-28/ROUND2.md)
- [2차 인계](records/2026-10-28/HANDOFF_ROUND2.md)
- [2차 집계](records/2026-10-28/round2_metrics.json)
- [1차 전체 실험](records/2026-10-28/README.md)
- [1차 인계](records/2026-10-28/HANDOFF.md)
- [1차 지표](records/2026-10-28/metrics.json)
- [당시 모델 목록](records/2026-10-28/model_inventory.json)
- [당시 원시 파일 해시](records/2026-10-28/source_index.json)
- [공개본의 결과 해시](records/public_source_index.json)
- [재현 안내](docs/REPRODUCING.md)

과거 기록은 당시의 결론을 보존합니다. 최신 선택과 상태는7차 기록과round7_state.json이 우선합니다. F1200·엔진v2·CPU1·실제8회를 유지하며 native2배속 품질은 미해결입니다. 모든 기존 시험 세트는 이미 사용되었으며, 후속 독립 검증에는 새 시험이 필요합니다.
