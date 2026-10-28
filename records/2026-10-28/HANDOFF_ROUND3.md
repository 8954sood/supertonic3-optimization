# 최신 인계 — 공개 저장소 마무리

훈련과 서버 측정은 종료했다. 새 실험을 진행 중이라고 가정하지 않는다. 현재 사용자가 청취한 기준은 `models/cfg-mixed-v1-selected`이며 더 작은 모델로 교체하지 않았다. 가중치·훈련 데이터·WAV는 로컬에 보존하며 Git에 넣지 않았다. 공개 저장소는 소스와 측정 기록이다.

## 확정 결과

- 선택: 혼합 폭 단일 경로 CFG 증류, `distill/joint-cfg-mixed-v1/best_step_00600.pt` 출발. 원본 CFG4의 근사이며 같은 수학적 CFG4가 아니다. 실제 8회, 지원 steps=[8]. 반복부50,369,776파라미터, frontend/decoder FP32.
- EPYC CPU1, 각3회 중앙값 KO5.822899초/EN5.458231초, peak표본393.994MB. CPU/wall 약1.0. KO5.320초/EN4.934초 음성으로 아직 실시간보다 약간 느리다.
- 폭1024 단일 경로2400회: KO4.707243초/EN4.392192초, peak표본388.256MB, CPU/wall 약1.0. 위 두 문장은 실시간보다 빠르지만 음질 우선 선택을 대체하지 않았다.
- 원래CFG4 혼합 폭 선택400회: KO9.994839초/EN9.095747초, peak356.917MB. 더 보수적인 대조군으로 보존.
- 새시험12문장: 원본/CFG4혼합폭/CFG증류혼합폭/CFG증류1024의 KO CER는 각각5/139,5/139,5/139,8/139. EN WER는0/69,0/69,3/69,0/69.
- CFG증류혼합폭 새시험 정렬 logmel 평균차이3.943dB. 객관적 음질/MOS가 아니다.
- 사용자는 선택 후보의 한국어 M1·seed42·8스텝 실제서버 샘플을 원본과 비교하고 거의 차이가 없다고 답했다. 한 샘플 청취 의견이며 EN 시험오류나 전체언어 검증을 덮어쓰지 않는다.

원시: `results/epyc_cfg_selected_v1_8.json`, `results/guard_cfg_selected_v1.json`, `results/epyc_cfg1024_final_v1_8.json`, `results/guard_cfg1024_final_v1.json`, `results/distill_cfg_selected_test_v3.json`, `results/distill_cfg_controls_test_v3.json`, `results/acoustic_cfg_selected_test_v3.json`.

## 선택하지 않은 실험

2400회 혼합 폭 최종과200회8스텝 음성 rollout은 latent 손실이 감소해도 validation 음향이600회 선택보다 나빴다. 단일경로1024는 발음 오류 증가. decoder1024 공동학습800회는 상대파형MSE0.276이 남아 채택하지 않았다. decoder fake QAT와 실제ORT의 상대RMSE0.222로 QAT도 신뢰할 수 없었다. 같은 가중치의 문장 조건 캐시는 실제서버에서 느려져 제외했다. 자체 C/BLAS 커널도 표준ORT보다 느렸다. 낮은 rank 진단만 수행했으며 low-rank 학생은 구현하지 않았다.

## 파일과 재현

`docs/models/`에 주요3개 모델의 manifest와 ONNX 식별 해시를 보존했다. 실제 가중치는 Git에 없다. `docs/REPRODUCING.md`의 과정으로 새 모델을 훈련할 수 있으나 바이트 단위 재현을 보장하지 않는다. 과거CFG4선택400회 FP32 checkpoint는 당시 best.pt가1200회로 바뀌어 없어졌으며400회 ONNX만 로컬에 보존했다. 현재CFG증류600회 checkpoint는 별도 파일로 보존했다.

원본 core와 런타임의 KO/EN8스텝 완전일치 검사는 통과했다. FP32캐시4조합도 기존 통과. 모델없는 공개 CI는 소스 컴파일·공개 파일 검사만 수행한다. 모델 추론 시험과 혼동하지 않는다. ARM 미측정, Docker 미검증, 문장 내부 스트리밍 미구현이다.

## 이어서 할 일

새 고유 train 문장과 독립 test를 늘리고 EN의3/69오류를 우선 확인한다. 기존 `corpus.json` test와 `corpus_cfg_v3.json` test는 모두 소비했다. 새 모델 선택에 재사용하면 독립 시험이 아니다. 표본을 늘린 다화자·장문 청취가 필요하다. 실제 ARM에서 원본/후보 해시·출력·시간을 확인한다. 더 작은 모델을 채택할 때는 청취 기준 후보와 비교해야 한다.

접속 정보는 공개 기록에서 제거했다. 인증·서버 설정은 로컬의 별도 비공개 자료로 관리하며, 공개 저장소만으로 원래 서버에 접속할 수 없다.
