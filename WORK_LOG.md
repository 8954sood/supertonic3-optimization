# 작업 기록 색인

- [14차: 16행·계층적 잔여 처리·직전 대비 HTTP 3.82% 감소·96쌍 바이트 일치](records/2026-10-30/ROUND14_ENGINE.md)
- [14차 상태·타일 후보·고정 13차 라이브러리 비교·재현](records/2026-10-30/round14_state.json)

- [13차: 실패 후보 4종·행 병렬 정규화 채택·직전 대비 HTTP 9.04% 감소·파형 보존](records/2026-10-30/ROUND13_ENGINE.md)
- [13차 상태·재현 명령·빌드·호환 검사·남은 과제](records/2026-10-30/round13_state.json)

- [12차: 최신 main 동기화·표준 ORT 조건 캐시·파형 보존·실제 HTTP 교대 측정](records/2026-10-30/ROUND12_ENGINE.md)
- [12차 상태·재현 명령·남은 과제](records/2026-10-30/round12_state.json)

- [11차: 호환 서버·PCM/SSE 스트리밍·맥 로컬 측정·배속 입력·1.5배속 발음 피드백 정정](records/2026-10-29/ROUND11_SERVER.md)
- [11차 상태와 검사 집계](records/2026-10-29/round11_state.json)

- [10차: 커밋 후 엔진 후보5종 재검증·모두 미채택·v4 유지](records/2026-10-29/ROUND10_ENGINE.md)
- [10차 기계 판독 상태와 교대 측정 집계](records/2026-10-29/round10_state.json)

- [최신: 엔진v4·내부 배속 후보의 RTF·예전 가중치 정리](records/2026-10-29/ROUND9_ENGINE.md)
- [9차 상태·측정 집계](records/2026-10-29/round9_state.json)
- [선택2배속 패키지 Hugging Face 공개·기존 파일 보존·다운로드 검증](records/2026-10-29/PUBLICATION_RETIME.md)
- [최신: 모델 내부2배속·기계음 개선 청취·남은 발음 오류·실제 서버 검사](records/2026-10-29/ROUND8_NATIVE_SPEED.md)
- [8차 기계 판독 상태](records/2026-10-29/round8_state.json)
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

과거 기록은 당시의 결론을 보존합니다. 선택 엔진과 공개 모델은9차 기록, 서버·스트리밍 및 최신 청취 피드백은11차 기록, Mac 표준 ORT 조건 캐시는12차, 행 병렬 정규화와 최신 성능 비교는13차 기록이 우선합니다.10차 미채택 엔진 시도도 별도로 보존합니다. F1200·CPU1·실제8회를 유지하며 EPYC용 선택 엔진은v4입니다. 선택 실행하는 내부2배속 경로의 짧은 청취 예시는 개선됐지만 확대 검증에는 발음 오류가 남습니다. 사용자가 요청한 예전 가중치 정리 목록도9차에 보존합니다. 모든 기존 시험 세트는 이미 사용되었으며, 후속 독립 검증에는 새 시험이 필요합니다.
