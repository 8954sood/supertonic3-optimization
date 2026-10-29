> 최신 지침: [ROUND5_LOCAL.md](ROUND5_LOCAL.md). 추가 업로드·커밋·푸시는 요청 전까지 중지했다. 새1024-v2와 확대 데이터는 아직 공개되지 않았다. 아래 공개 자산 링크는 기존3개 모델을 가리킨다.

# 이어서 작업할 때

최신 결론은 [ROUND4.md](ROUND4.md), 수치는 [round4_metrics.json](round4_metrics.json)에 있다. **기본 모델은 cfg-mixed-v1-selected를 유지한다.** 새 `cfg1024-v2`는 공개 실험 후보다. 서버는 CPU1·실제8스텝을 유지하고, ARM 접속 정보를 다시 요청하지 않는다. 사용자가 새 후보의 ARM 검사를 나중에 직접 진행한다.

## 현재 자산

- 공개 코드: https://github.com/8954sood/supertonic3-optimization
- 모델/필수 초기·최종 가중치: https://huggingface.co/babihoba/supertonic3-optimization
- 교사 궤적: https://huggingface.co/datasets/babihoba/supertonic3-optimization-data
- 이전 모델의 완전 검증 고정 revision과 공개 재현 결과: [PUBLICATION.md](PUBLICATION.md)
- 모든 다운로드·재학습 명령: [REPRODUCING.md](../../docs/REPRODUCING.md)
- 사용자 ARM PR #1은 이전 선택 모델 측정이다. draft를 유지하고 main 충돌만 해결했다.

로컬에서 새 후보는 `models/round4-cfg1024-selected`, 교사 데이터는 `distill/trajectories-round4`, 초기 가중치는 `distill/joint-cfg-1024-v1/best.pt`, 최종은 `distill/joint-cfg1024-round4/best_step_02000.pt`이다. 공개 폴더 이름은 `models/cfg1024-v2`이다. 실제 벡터 해시는 `docs/models/cfg1024-v2.json`에 고정했다. 전체 경로를 바꾼 경우 `trajectory_path()`가 NPZ를 현재 --data 밑에서 찾는다.

## 이미 끝난 일과 다시 하면 안 되는 해석

조건 캐시·head INT8·FastGELU·head SVD를 조사했다. 반복부 캐시는 로컬 비트 일치였으나 서버에서 확실한 이득이 없었다. 근사 GELU도 기본 모델에 반영하지 않았다. 원래1,664개 NPZ의 업로드는 완료했고 전체 원격 해시·공개 다운로드·실제 학습1회·선택 체크포인트 재내보내기를 검증했다. 업로드 대기 상태로 오해하지 않는다.

확대 train80문장/240음색seed/1,920쌍, validation16문장/128쌍으로 이전1024를 추가2,000회 학습했다. dev와 음향 거리는 일부 좋아졌지만 새 독립 test의 KO CER는 이전6/218→새10/218로 나빠졌다. 그래서 새 모델은 기본 대체가 아니다. 기존 사용자 “거의 차이 없음” 의견은 이전 선택의 한국어 M1 샘플에만 해당한다.

모든 `corpus*.json`의 기존 test는 소비했다. `corpus_round4.json`과 `corpus_round4_train.json` test도 소비했다. 다음 모델은 train 확장·청취/개발 검증 후 다시 새 test를 고정해야 한다. 동일 텍스트의 음색만 바꿔 독립 test라고 부르지 않는다. test 악화를 보고 이번2000회 가중치를 다시 조정하지 않았다.

## 다음 권장 작업

1. 사용자가 새 후보와 이전 선택 모델을 같은 ARM 환경에서 측정하고 대표 음색을 청취한다. 기본은1스레드·실제8스텝이다. `experiments/paired_low_memory.py`로 두 모델을 번갈아 한 개씩 로드할 수 있다. 새 결과 이름을 써서 과거 측정을 덮지 않는다.
2. 모델을 더 줄이려면1,920개 궤적만 늘리는 것보다 다양한 고유 텍스트와 언어·음색·길이 분포를 늘리고, 실제8회 rollout의 음향 손실을 검토한다. 이미 소비한 test를 선택에 쓰지 않는다.
3. 새 보코더/DSP head는 별도 학습 과제다. 이번 실험에서 Paradee 구조를 그대로 이식하거나 PDN 엔진을 구현했다고 주장하지 않는다.
4. GPU/MPS 학습과 저메모리 CPU 추론 조건을 분리한다. 원본 FP32는1GB 서버에서 유효한 속도 기준을 얻지 못했다. 기존12초 INT8 기준을 FP32 원본 속도로 쓰지 않는다.

## 작업 보호

개인키·토큰·서버 접속 주소는 기록에 넣지 않는다. 공개 Git에는 가중치·WAV·NPZ·업로드 도구의 로컬 상태를 넣지 않는다. Hub에는 필수 모델/데이터만 올린다. 공유 서버에서는 기존 서비스를 건드리지 않고 순차 실행한다. 메모리 감시는50ms 소프트 표본 방식이며 OS 강제 상한이 아니다. 새 모델의 원본 일치율이나 MOS를 임의로 만들지 않는다.
