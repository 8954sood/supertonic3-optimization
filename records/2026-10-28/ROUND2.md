> 당시 시점의 기록입니다. 최신 확정 상태는 [HANDOFF_ROUND3.md](HANDOFF_ROUND3.md)를 참조하세요.

# 2차 최적화 — 음질 원인 분리와 전체 궤적 증류

사용자 요청: 원본보다 어색한 소리를 개선하면서 성능·WER/CER·일치와 시도 기록을 이어갈 것. 1차 스냅샷 README/metrics/source_index는 보존한다.

## 요구 조건

추론 CPU1, 실제 denoising8회 이상, 원래 CFG4. 실시간 목표는 미달. 훈련은 Mac MPS, 추론은 CPU. test12문장은 아직 사용하지 않음. 아래 비교는 validation4문장 KO2/EN2, M1/F1, seed400+case로 고정했다. 통계적 대표성이나 청취 품질 보장은 없다.

## 원인 분리 결과

원본과 후보를 같은 문장/음색/seed/8스텝으로 비교했다. 원본 길이 예측/텍스트 인코더/보코더를 고정한 부분 교체도 했다.

| 후보 | 정렬 log-mel MAE(dB, 낮을수록 원본에 가까움) | pYIN F0 중앙 절대 오차의 문장 평균(cents) |
| --- | ---: | ---: |
| 수학적 연산 보존 FP32캐시 | 0.00014 | 0 |
| 전체 INT8 | 2.382 | 10 |
| 반복부만 INT8 | 1.404 | 5 |
| 길이/텍스트만 INT8 | 0.762 | 0 |
| 보코더만 INT8 | 1.649 | 0 |
| 기존1024 전체INT8 | 5.481 | 45 |
| 기존1024 반복부FP32, 나머지 원본 | 5.004 | 50 |
| 기존1024 반복부INT8, 나머지 원본 | 5.080 | 47.5 |
| 기존1536 반복부INT8, 나머지 원본 | 3.364 | 23.75 |

해석: 1024의 어색함을 INT8 하나로 설명할 수 없다. FP32 학생도 원본과 큰 차이가 있어 층별 독립 증류의 누적 오차가 의심된다. 보코더 정수화 자체도 스펙트럼에 차이를 만든다. 사용자 청취 파일명은 특정되지 않았으므로 특정 모델이 그 청취 대상이었다고 단정하지 않는다.

원시 결과: `results/quality_ablation_v2.json`, `results/acoustic_ablation_v2.json`, `results/quality_components_v2.json`, `results/acoustic_components_v2.json`. WAV: 각 JSON의 audio/reference_audio.

정의: 16kHz 분석, 1024 FFT/160hop, 64mel 60~7600Hz, 전역1e-9 power floor, MFCC1..19 Euclidean DTW(15% band), 활성 프레임 비교. pYIN65~500Hz, 양쪽 유성 구간만 pitch cents. 길이 오차와 유무성 일치도 별도 기록. DTW가 시간 차이를 흡수하므로 길이 지표도 함께 읽는다. MOS나 화자 유사도 점수가 아니다. pYIN 추정 자체의 오류 가능성도 있다. 사용자 인용 Paradee의 배음비 수치와 같은 측정법이라고 주장하지 않는다.

## 새 구현과 검증

- `distill/torch_graph.py`: 원본 ONNX의 미분 가능한 훈련용 구현. 운영 추론 엔진이 아님.
- 반복부 원본 ORT 대조: 상대RMSE3.78e-7, 최대오차2.15e-6 (`results/torch_graph_parity.json`). 현재 한 실제 입력으로 확인했으며 다양한 입력으로 확대 필요.
- 보코더 대조: 상대RMSE9.69e-6, 입력 latent에 대한 유한한 기울기 확인 (`results/torch_vocoder_parity.json`). 역시 한 입력 검사.
- `distill/collect_vectors.py`: 교사 원본8스텝 전체 입력/출력 수집. 훈련40문장320쌍, 검증8문장64쌍. `distill/trajectories-v1` 약67MB. test미사용.
- `distill/train_joint.py`: 28개 FFN을 전체 CFG4 반복부를 통과시켜 공동 학습. 입력을 지배하는 잔차를 제외한 교사 update 정규화 MSE. 첫1024 실행600회: 검증0.10933→최선약0.07646(400회), 마지막600회는0.08081. 최선 저장.
- 위 공동학습1024를 INT8 내보내고 나머지 원본FP32를 사용한 후보의 logmel 차이는 네 문장 약4.63dB로 이전5.08dB보다 줄었으나 여전히 큼. `results/acoustic_joint1024_v1.json` 참조.
- `distill/train_rollout.py`: 학생이 실제8회 연결 추론, 원본 보코더 고정, 최종 latent MSE와 512/1024/2048 FFT의 spectral convergence+log-magnitude L1을 공동 학습. 학습 벡터장의 모든 학습 가중치를 열고 원본 보코더는 고정. 복소 FFT는 CPU로 옮겨 미분한다. 새로운 DSP헤드·적대적 학습은 아직 없음.
- 첫1024 rollout200회: 검증4문장 손실0.44838→최선0.35161(75회), latent상대MSE0.26454→0.17958, MRSTFT1.22555→1.14691. 이후 일부 악화해 best를 선택. 과적합/데이터 부족 가능성. 실제 음향/발음 비교가 끝나기 전 채택하지 않는다.

## 진행 상태

업데이트 중. 후속 측정은 아래에 추가한다. 원시 JSON/로그/체크포인트를 덮어쓰지 않고 버전 이름을 사용한다.

### 추가 시도와 실패

- 원본 반복부 8입력(훈련/검증·M/F·초기/마지막 단계) 및 보코더4입력 대조 확대: 반복부 상대RMSE 약3.8e-7~2.1e-6, 보코더 약7.0e-6~9.7e-6. `results/torch_graph_parity_suite.json`. PyTorch 훈련용 구현의 초기 수치 대조를 통과했으며 모든 입력에 대한 보장을 의미하지 않는다.
- 1024의 waveform rollout 후보는 음향 차이가 약4.57dB로 여전히 크고, 선택4문장 ASR에서 한국어 오류4개가 추가돼 채택하지 않았다. `results/asr_rollout1024_v1.json`.
- 1536 공동학습1000회: 교사 업데이트 MSE가 약0.0315 수준으로 내려갔다. 그 뒤8스텝 waveform rollout200회는 최선이 **초기 step0**이었다. 실제 출력이 좋아진 것으로 기록하면 안 된다. 내보낸 rollout1536-v1은 joint1536-v1과 같은 best 가중치다. 선택4문장 ASR KO1문자 오류, EN0. `results/asr_rollout1536_v1.json`.
- `distill/trajectories-v2`: 같은 훈련40문장을 문장별5음색/seed로 확장, 총200개 문장·음색·seed 조합, 1600단계쌍. **새로운 고유 텍스트200개가 아니다.** 검증8문장64쌍은 유지. test12 미사용. 약289MB.
- 위200조합을 사용한1536 rollout400회는 현재까지 초기 best를 넘지 못하고 있다. 끝난 뒤 최선 checkpoint step을 확인할 것.
- 보코더 QAT를 준비하면서 fake quant와 실제 ORT reduced INT8 대조를 했으나 출력 상대RMSE가0.222로 컸다. `results/qat_decoder_parity.json`. 현재 fake QAT를 정밀한 실행 모사로 간주하지 않고 보코더 QAT 훈련은 시작하지 않았다. 이전 층별 QAT도 실제 ONNX 음성 검사를 별도로 해야 한다.
- 혼합 폭 후보: 층별 검증 기준으로512(오차<=.01),1024(<=.01),1536(<=.15), 그 외 원본2048을 선택. 선택은 `distill/mixed-v1/selection.json`. 민감한 층은 원본을 보존하며200조합의1600단계쌍으로 전체 반복부 공동학습 중. 초기 업데이트 검증MSE0.02163.

현재 실행·결과 파일은 `round2_metrics.json`에도 취합한다. 1차스냅샷의59개원시JSON 해시 및 metrics는 덮어쓰지 않았다.

### 혼합 폭 중간 결과

`models/mixed-v1-preview-int8`는 선택4문장 logmel MAE 평균2.754dB, 문장별2.624/2.548/2.874/2.971dB, F0 오차의 평균20cents였다. 기존1024 전체INT8의5.481dB/45cents보다 원본에 가까워졌다. 다만 preview는 원본FP32 frontend/decoder를 사용하므로, frontend/decoder도 복원한 기존1024 반복부INT8의5.080dB/47.5cents와 비교하면 축소 방식 변경만으로도 차이가 줄었다. ASR KO2/EN2 모두 오류0. 이4문장은 후보 선택용 validation이며 청취 품질이나 장문/다른 음색 보장은 아니다. 원시 파일 `results/acoustic_mixed_preview_v1.json`, `results/asr_mixed_preview_v1.json`.

혼합 폭1200회 공동학습 완료: 업데이트 검증 MSE0.0216288→0.0182932(최선1200회). 아래 최종 모델은 같은 가중치를 사용하되 quality는 frontend/decoder를 FP32 MatMul로, fast는 기존 전체INT8 구성으로 비교한다. 최종 음향·ASR·실제 서버 결과를 이어 적는다.

### 후보 고정과 선택 근거

- 혼합 폭 분포: 512폭2층, 1024폭4층, 1536폭12층, 원본2048폭10층. 원본 대비 반복부 파라미터 감소는 약20%로, 반폭1024 모델보다 보수적이다.
- 1200회 최종 가중치는 교사 업데이트 MSE가 가장 작았으나 실제 validation 음성은 logmel 약2.97dB 및 KO1문자 오류로, 중간400회 내보내기의2.754dB/선택4문장오류0보다 좋지 않았다. 따라서 업데이트 MSE만으로 선택하지 않고 **400회 ONNX 내보내기**를 선택했다.
- `models/mixed-v1-selected`: 위400회 반복부INT8를 보존하고 frontend/decoder는 수학적 연산을 보존한 FP32 MatMul로 실행. 원본 CFG4, 실제8스텝 유지. 네 문장 logmel2.754dB, F0 평균20cents, ASR오류0이 그대로 재현됐다.
- 더 빠른 전체INT8 구성인 `models/mixed-v1-fast`(1200회)는 logmel약3.68dB/KO1문자 오류로 선택하지 않았다. 아직 서버에 올린 파일이 있다는 이유로 기본 후보로 취급하면 안 된다.
- **체크포인트 보존 문제와 수정:** 400회 당시의 `best.pt`는 이후1200회로 갱신되었고 별도 FP32 step400 체크포인트를 보존하지 못했다. 그러나 그때 내보낸 Q8 ONNX는 독립 파일로 남아 있어 그대로 복사·해시로 보존했다. 선택 모델 manifest의 vector_sha256과 source_step400을 기준으로 사용한다. 향후 train_joint/train_rollout은 best 갱신마다 `best_step_XXXXX.pt` 하드링크를 남기도록 수정했다. 과거 checkpoint 경로만 보고 같은 가중치라고 가정하면 안 된다.
- 확대 validation8문장·6음색: 원본 KO2/88(2.27%), 선택 후보 KO1/88(1.14%), 둘 다 EN0/46. 작은 ASR 차이이며 자연스러움 우위를 의미하지 않는다. `results/distill_mixed_selected_validation_v1.json`.
- 이 시점에서 후보를 고정하고 처음으로 별도 test12문장을 실행한다. 이후 이 test를 튜닝용으로 다시 사용하지 않는다. 향후 추가 튜닝에는 새로운 시험 세트를 마련해야 한다.

## 2차 최종 측정과 현재 후보

현재 선택 모델: **`models/mixed-v1-selected`**. 400회 시점 Q8 반복부 + 수학적 연산을 보존한 FP32 frontend/decoder. 원래 CFG4, denoising8회, CPU1. 원본과 청취상 같다고 승인한 모델이 아니다.

| 실제 EPYC 1GB 서버 | KO 초 | EN 초 | 표본 최대RSS | 판단 |
| --- | ---: | ---: | ---: | --- |
| 초기 전체INT8 | 12.117 | 11.802 | 322MB | CPU별 출력 차이가 큼; 구형x86 음질 기준으로 재검토 |
| 이전1024 전체INT8 | 7.471 | 7.398 | 259MB | 빠르지만 음질 문제로 보류 |
| **혼합 폭 선택 후보** | **9.995** | **9.096** | **356.9MB** | 현재 속도·품질 절충 후보 |
| 같은 가중치+조건 캐시 | 11.215 | 11.918 | 382.0MB | 느리고 메모리도 증가해 제외 |

모두 8스텝. 선택/캐시는 워밍업 후 각3회 중앙값. 서버는 공유 상태이며 cache의 CPU/wall 약0.93~0.99로 일부 경합도 있었다. 캐시는 Mac4조합에서 비트 일치했지만 실제 서버에서 속도 이득이 없어 채택하지 않았다. 모델 크기는 선택 패키지 약188MiB. 최초 전체INT8 대비 소요 시간 KO17.5%/EN22.9% 감소지만, 이전1024 후보보다는 느리고 메모리도 더 사용한다. 실시간 목표는 아직 미달이다.

원시: `results/epyc_mixed_selected_v1_8.json`, `results/guard_mixed_selected_v1.json`, `results/epyc_mixed_cached_v1_8.json`, `results/guard_mixed_cached_v1.json`, `results/equivalence_mixed_cache_v1.json`. 선택 후보는 가용RAM 최소282MB를 남겼고 감시기 중단 없이 완료됐다.

### 독립12문장 시험(최초 사용)

| 모델 | KO CER | EN WER |
| --- | ---: | ---: |
| 원본 FP32 | 4/142 = 2.82% | 0/72 = 0% |
| 혼합 폭 선택 후보 | 4/142 = 2.82% | 1/72 = 1.39% |

음색6종, 고정seed400+case, KO6/EN6, steps8. 원본과 같은 CER이 소리 일치를 의미하지 않는다. EN에서1단어 추가 오류가 남았으므로 완전한 동등성으로 기록하지 않는다. 시험12문장 평균 logmel차이3.256dB, F0 문장별중앙절대오차 평균26.67cents, 유무성 일치0.926. `results/distill_mixed_selected_test_v1.json`, `results/acoustic_mixed_test_v1.json`. 이 시험을 읽고 모델을 추가로 튜닝하지 않았다. 후속 튜닝에는 새 독립 시험 세트가 필요하다.

### 실제 구형CPU 출력에서 확인한 음향 차이

같은 benchmark두문장·M1·seed42·8스텝의 **서버가 만든 WAV**를 원본FP32 Mac reference와 비교했다. 원본FP32 자체는 서버 메모리 제한으로 실행하지 못했으므로 reference가 다른 CPU라는 한계가 있다.

| 서버 후보 | 원본 대비 정렬 logmel MAE 평균(dB) | F0 문장별중앙절대오차 평균(cents) |
| --- | ---: | ---: |
| 초기 전체INT8 | 15.559 | 85 |
| 이전1024 전체INT8 | 16.402 | 105 |
| **혼합 폭 선택 후보** | **3.291** | **30** |

2문장에 한정한 음향 대리 지표다. “음질이80%좋아졌다” 같은 지각적 비율로 환산하면 안 된다. 서버 후보의 벤치마크 두문장 ASR는 오류0이었다. 실제 자연스러움은 청취 확인이 남아 있다. `results/acoustic_server_original_mixed_v1.json`, `results/asr_epyc_mixed_selected_v1.json`. WAV와 reference경로는 원시JSON에 있다.

### CPU별 INT8 출력 차이 조사

Mac/EPYC의 전체INT8 및 선택 모델 각4모듈 SHA-256이 모두 같은 것을 확인했다. ORT1.31.1·동일 text/voice/seed/steps로 비교했다.

- 초기 전체INT8은 서버의 길이가 Mac보다 KO3.18%, EN8.13% 길었다. Mac/서버 서로의 logmel차이는 KO15.918/EN14.449dB였다.
- 선택 후보는 길이가 정확히 같고, Mac/서버 서로의 logmel차이는 KO1.786/EN2.061dB, F0중앙오차각10cents였다. 다만 파형은 비트 동일하지 않으며 상관은 약0.63/0.60이다. “CPU가 달라도 완전히 같은 소리”로 주장하지 않는다.
- `results/crossplatform_mixed_v1.json`, `results/acoustic_crossplatform_mixed_v1.json`, `results/model_hashes_crossplatform_mixed_v1.json` 참조.
- 구형 AVX2에서 U8S8 정수 곱의 포화가 원인일 가능성이 있다. ORT 공식 문서도 VPMADDUBSW 포화 및 reduce_range/U8U8 완화를 설명한다. **이번관찰만으로 그 원인을 단독 확정한 것은 아니다.** 선택 모델은 반복부reduce_range와 frontend/decoder FP32를 동시에 적용했기 때문에 각 변경을 분리한 추가대조가 필요하다. [ORT 공식 설명](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

### 실행·아티팩트

```bash
.venv/bin/python synthesize.py '안녕하세요. 기존 음성과 비교해 주세요.' --model models/mixed-v1-selected --lang ko --voice M1 --steps 8 --output samples/mixed_selected.wav
```

`dist/mixed-v1-quality-runtime.tar.gz`는 선택모델·원본라이선스·실행기·고정의존성·간단한설명을 포함한 독립패키지다. 개인키·훈련데이터·다른모델·가상환경은 포함하지 않는다. 배포 품질 승인을 뜻하지 않는다. SHA-256파일도 함께 저장한다.

### 추가 원인 대조

기존 서버 profiling실험의 `fast-avx2-int8` 한국어WAV를 가져와 Mac의 같은모델/입력과 비교했다. 4모듈해시가 모두 일치했다(`results/model_hashes_avx2_mixed_v1.json`). reduce_range를 전체에 켜면 Mac/서버 길이차이는 약0.14%로 줄고 F0중앙오차는10cents였지만, logmel차이는 여전히10.031dB로 컸다. 원본FP32 reference 대비 서버 logmel차이는15.601dB였다. 한 문장·기존 profiling WAV의 대조이며 reduce_range만으로 모든 음향문제가 해결됐다고 판단할 수 없다. `results/acoustic_reducedrange_mixed_v1.json`.

따라서 구형CPU의 INT8실행·특히보코더가 추가조사 대상이다. 선택후보가 잘나온 결과를 포화완화 하나의 효과로 귀속하지 않는다. frontend/decoder FP32복원과 혼합폭학생 모두 포함된 결과다.

혼합반복부의 정밀한 FP32파라미터수는 원본64,014,576→50,369,776(21.3%감소)이다. Q8 ONNX파일크기와 파라미터개수를 혼동하지 않는다.
