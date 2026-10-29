# Oracle ARM 선택 모델 실측 — 2026-10-29

서버의 `~/code/tts`에서 현재 청취 기준 모델 `cfg-mixed-v1-selected`를 측정했다. 모델과 추론 로직은 변경하지 않았다. ARM Neoverse-N1 2코어, RAM 12,512,555,008 bytes, Ubuntu 22.04 / Linux 6.8 aarch64. 다른 서비스가 실행 중인 서버에서 순차 측정했다.

## 조건

- 소스 커밋: `36423c3aac887db1158caba41f7516a620463012`. 추가 실험 실행기는 `experiments/benchmark_ort_threads.py`.
- 모델: `babihoba/supertonic3-optimization`, 고정 revision `85bc100ca8d7cefad17b63a4c7ba489638dbe2f2`.
- 배포 manifest에 있는 선택 모델 파일 21개 모두 크기·SHA-256 검증 통과. 반복부 해시 `d2b63c7a648b93b1d1d94a4443a47bd92ec0aca75299a27dffb6473bc34b93ea`.
- Python 3.12.16, ONNX Runtime 1.31.1, NumPy 2.5.4, CPUExecutionProvider.
- M1, seed 42, speed 1.05, 실제 denoising 8회. 문장별 1회 워밍업 뒤 5회 중앙값.
- CPU1: 코어 0 고정, ORT intra/inter-op 및 BLAS 1스레드. CPU2: 허용 코어 [0, 1], ORT intra-op 2, inter-op 및 BLAS 1스레드.
- RTF = 합성 시간 / 음성 길이. 낮을수록 빠르며 1 미만이면 실시간보다 빠르다. 모델 로딩·WAV 저장은 제외한다.

## 결과

| 문장 | 음성 길이 | CPU1 합성 | CPU1 RTF | CPU2 합성 | CPU2 RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 한국어 짧음 | 5.320초 | 1.468초 | 0.276 | 0.841초 | 0.158 |
| 한국어 장문 | 17.616초 | 4.960초 | 0.282 | 2.692초 | 0.153 |
| 영어 짧음 | 4.934초 | 1.369초 | 0.278 | 0.793초 | 0.161 |
| 영어 장문 | 16.909초 | 4.791초 | 0.283 | 2.591초 | 0.153 |

CPU1 약 3.53~3.62배 실시간, CPU2 약 6.22~6.54배 실시간이다. CPU2는 CPU1보다 약 1.73~1.85배 빠르다. CPU/wall은 각각 약 0.99~1.00과 1.80~1.87로, 두 코어 사용을 확인했다.

모델 로딩은 CPU1 1.245초, CPU2 1.221초. 각 실행의 50ms 표본 최대 RSS는 448.5MB / 442.7MB. 시스템 가용 메모리는 두 실행 모두 9GB 이상이었다. 이 로딩값은 디스크 캐시를 비운 콜드 스타트 측정이 아니다.

네 문장의 저장된 PCM16 WAV는 CPU1과 CPU2가 샘플 단위로 동일했다(RMSE=0). 길이·샘플레이트 일치, 유한한 출력, 클리핑 0개를 확인했다. 원본 모델과의 음질 동등성이나 ARM/x86 음질 동등성을 새로 검증한 결과는 아니다. ASR·MOS·다화자 평가는 수행하지 않았다.

## 재실행

서버 `~/code/tts`에서:

```bash
.venv/bin/python benchmark.py \
  --model .hf-assets/model/models/cfg-mixed-v1-selected \
  --steps 8 --repeats 5 --affinity 0 \
  --output results/arm_selected_cpu1_20261029.json

.venv/bin/python experiments/benchmark_ort_threads.py --ort-threads 2 \
  --model .hf-assets/model/models/cfg-mixed-v1-selected \
  --steps 8 --repeats 5 \
  --output results/arm_selected_cpu2_20261029.json
```

두 명령은 순차 실행한다. 실제 실험에는 `memory_guard.py --max-rss-mb 4000 --reserve-mb 2000 --seconds 1200` 감시기도 적용했다. 표본 감시이며 OS 하드 제한이 아니다. 기본 `runtime.Engine`은 계속 1스레드이므로, 위 2스레드 결과는 추가 실행기로만 재현된다. 문장 내부 스트리밍은 없다.

## 산출물

- 원시 시간·단계별 시간: `results/arm_selected_cpu{1,2}_20261029.json`.
- 메모리 감시: `results/arm_selected_cpu{1,2}_guard_20261029.json`.
- 모델 provenance·해시: `results/arm_model_revision_20261029.json`, `results/arm_verified_assets_20261029.json`.
- WAV 비교: `results/arm_audio_checks_20261029.json`.
- 환경·의존성: 이 문서와 같은 폴더의 `arm_environment_20261029.txt`, `arm_dependencies_20261029.txt`.
- 음성: 로컬과 서버의 `samples/arm_selected_cpu{1,2}_20261029/*.wav` (Git 제외).

단일 요청 지연 기준 이 서버에서는 2스레드가 유리했다. 동시 요청 처리량·지속 부하·다른 서비스 부하가 큰 상황은 별도 측정이 필요하다.
