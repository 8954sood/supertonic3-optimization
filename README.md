# supertonic3-optimization

[Supertonic 3](https://huggingface.co/Supertone/supertonic-3)의 **CPU 한 스레드·실제 denoising 8회** 최적화 실험입니다. 그래프 변환, 문장 조건 계산 재사용, INT8, 혼합 폭 증류, CFG 교사 출력의 단일 경로 증류를 비교했습니다. 추론은 CPU만 사용하고 증류 훈련은 Mac MPS에서 수행했습니다.

**소스와 측정 기록을 공개하는 저장소입니다.** 모델 가중치·체크포인트·수집 데이터·음성·배포 압축 파일은 포함하지 않습니다. 클론만으로 학습된 선택 모델을 실행할 수는 없습니다. 기본 그래프 최적화는 원본 다운로드 후 재현할 수 있고, 학습된 학생 모델은 아래 Hugging Face 링크에서 받을 수 있습니다.

## 현재 결과

실측 환경은 Oracle x86 **AMD EPYC 7551, 물리 1코어·논리 2CPU, RAM 약 954MB**입니다. 논리 CPU 하나에 고정하고 ORT/BLAS 연산 스레드를 1로 제한했습니다. 워밍업 후 KO/EN 짧은 문장 각각의 전체 파형 생성 시간 중앙값이며, 모델 로딩과 파일 저장은 제외합니다. 서버는 다른 작업과 공유했습니다. **Oracle ARM Neoverse-N1 2코어·12GB 실측은 [ARM 측정 PR #1](https://github.com/8954sood/supertonic3-optimization/pull/1)에 있습니다. 선택 모델의 한 스레드 RTF는 0.276~0.283, 두 스레드는 0.153~0.161이며, 기본 추론 설정은 계속 한 스레드입니다.**

| 후보 | 한국어 | 영어 | 표본 최대 RSS | 조건 |
| --- | ---: | ---: | ---: | --- |
| 전체 INT8 기준 | 12.12초 | 11.80초 | 약 322MB | 원래 CFG4, 2회 중앙값 |
| 혼합 폭, CFG4 보존 | 9.99초 | 9.10초 | 357MB | 3회 중앙값 |
| **혼합 폭, CFG 증류 선택 후보** | **5.82초** | **5.46초** | **394MB** | **3회 중앙값, 청취 기준 후보** |
| 폭 1024, CFG 증류 | 4.71초 | 4.39초 | 388MB | 3회 중앙값, 한국어 오류 증가 |

모든 후보는 8회를 실제 실행합니다. CFG 증류는 **원본 CFG4 업데이트를 한 조건 경로가 학습한 근사**이며 원래 CFG4와 동일한 계산이 아닙니다. 해당 학생의 지원 스텝은 8로 제한합니다. 작은 후보는 약 5.32초/4.93초 음성을 실시간보다 빨리 생성했지만, 청취 기준 후보를 대체하지 않았습니다. 선택 후보는 아직 실시간보다 약간 느립니다. 12초 기준도 이미 양자화된 모델이며, 원본 FP32는 메모리 감시기에 중단되어 유효한 서버 속도 비교값이 없습니다.

고정된 새 시험 12문장(KO/EN 각 6개), Whisper base INT8·CPU1 대리 검사:

| 모델 | 한국어 CER | 영어 WER |
| --- | ---: | ---: |
| 원본 FP32 | 5/139 (3.60%) | 0/69 (0%) |
| 혼합 폭 CFG4 | 5/139 (3.60%) | 0/69 (0%) |
| 혼합 폭 CFG 증류 | 5/139 (3.60%) | 3/69 (4.35%) |
| 폭 1024 CFG 증류 | 8/139 (5.76%) | 0/69 (0%) |

사용자는 선택 후보의 실제 서버 한국어 M1 샘플을 원본과 비교하고 거의 차이가 없다고 평가했습니다. 한 문장의 의견이며 모든 음색·언어·장문의 동등성이나 MOS 검증은 아닙니다. CER/WER를 음질이나 원본과의 일치율로 해석하지 않습니다. 선택 후보의 시험 12문장 정렬 logmel 평균 차이는 3.943dB입니다. [선택 후보 성능](results/epyc_cfg_selected_v1_8.json), [작은 후보 성능](results/epyc_cfg1024_final_v1_8.json), [선택 후보 ASR](results/distill_cfg_selected_test_v3.json), [같은 시험의 대조군](results/distill_cfg_controls_test_v3.json).

## 4차 추가 평가와 경량 실험

**기본 선택 모델은 그대로 유지합니다.** 추가 학습한 실험용 `cfg1024-v2`는 1코어·1GB에서 KO4.809초/EN4.373초였고, 같은 세션의 기본 선택 모델은5.911초/5.611초였습니다. 교차3쌍의 가속비 중앙값은1.247배/1.292배, 새 후보 표본 peak RSS395.58MB입니다. 이전1024 모델 대비 속도 개선을 입증한 결과는 아닙니다.

새 독립16문장에서 한국어 CER는 원본7/218, 기본 선택9/218, 이전1024 6/218, 새 후보10/218이었습니다. 영어는 모두2/109이며 같은 `streetlights`/`street lights` 구분에서 생겼습니다. 새 후보의 원본 대비 logmel 거리는 이전1024의4.813→4.460dB로 줄었지만, 한국어 발음 대리 점수는 나빠졌습니다. 새 모델의 청취·ARM 평가는 아직 없습니다. [전체 시도와 한계](records/2026-10-29/ROUND4.md), [원시 집계](records/2026-10-29/round4_metrics.json), [최신 인계](records/2026-10-29/HANDOFF_ROUND4.md)를 참고하세요.

조건 캐시·보코더 head INT8·FastGELU도 실제 서버에서 비교했으나 기본 모델을 바꿀 만한 일관된 이득은 확인하지 못했습니다. 새로운 독립 시험도 이제 소비했으므로 후속 학습 선택에 재사용하지 않습니다.

## 기본 최적화 실행

Python 3.12 이상, 모델 다운로드·변환을 위한 충분한 RAM과 디스크가 필요합니다. **아래 준비·변환 작업은 1GB 서버용이 아닙니다.** 메모리가 넉넉한 개발 환경에서 준비한 뒤 완성된 모델을 복사하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download.py
python prepare.py --mode fp32-cached --output models/fast-fp32-cached
python synthesize.py '안녕하세요. 하나의 스레드로 음성을 만듭니다.' \
  --model models/fast-fp32-cached --lang ko --steps 8 --output samples/hello.wav
```

`uv` 환경에서는 `uv pip install --python .venv/bin/python -r requirements.txt`로 설치할 수 있습니다. 고정 원본 리비전은 `3cadd1ee6394adea1bd021217a0e650ede09a323`입니다. `prepare.py`는 기존 출력 폴더를 덮어쓰지 않습니다.

- `fp32-cached`: 1×1 Conv 행렬곱 변환과 문장 조건 캐시. 원래 CFG4를 유지합니다. M4 Max CPU1 교차 측정에서 원본보다 4.54~4.71배 빨랐습니다. 이 수치를 EPYC나 ARM으로 환산하지 않습니다.
- `denoiser-int8 --reduce-range`: 반복부만 감소 범위 INT8로 변환합니다. 손실 변환이며 CPU마다 출력 차이를 확인해야 합니다.
- 원본 실행기는 공식 core와 KO/EN 8스텝 출력이 완전히 일치했습니다. FP32 캐시 후보의 4개 조합 파형 RMSE는 6.8e-7~5.8e-6이었습니다.

```bash
python tests_runtime.py
python tests_optimized.py
python benchmark.py --model models/fast-fp32-cached --steps 8 --repeats 3 \
  --cases ko_short en_short --output results/local_fp32_8.json
```

반복 요청에는 `runtime.Engine`을 한 번 로드하고 재사용하며 요청은 순서대로 처리합니다. 문장 내부 스트리밍은 구현하지 않았습니다. 원본 계열은 8~50스텝, CFG 증류 학생은 8스텝만 허용합니다. 학습된 모델을 별도 준비한 경우 `--model models/cfg-mixed-v1-selected`로 실행합니다.

## 재현과 후속 작업

학습된 모델과 중간 파일은 공개 Hugging Face 저장소에서 내려받습니다: [모델과 체크포인트](https://huggingface.co/babihoba/supertonic3-optimization), [증류 훈련 데이터](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data). Hugging Face 저장소에는 선택 후보와 비교용 두 모델, 재현용 초기/600회 체크포인트, 1,600개 학습·64개 검증 궤적이 있습니다. 저장소 용량은 약 1.2GB입니다.

```bash
# 선택한 모델만 다운로드(약 188MB)하고 파일별 SHA-256 확인
python download_artifacts.py --purpose inference
# 초기/600회 체크포인트와 훈련 궤적 다운로드(약 646MB)
python download_artifacts.py --purpose reproduce
```

```bash
python synthesize.py '안녕하세요.' --model .hf-assets/model/models/cfg-mixed-v1-selected \
  --lang ko --steps 8 --output samples/hello.wav
```

체크포인트·데이터로 학습을 다시 돌리는 방법은 [다운로드 및 학습 재현 안내](docs/REPRODUCING.md)에 있습니다. 두 저장소는 공개이므로 다운로드에는 토큰이 필요하지 않습니다.

- [현재 상태와 인계](records/2026-10-29/HANDOFF_ROUND4.md): 모델 선택, 실패, 시험 사용 이력, 다음 우선순위.
- [학습 재현](docs/REPRODUCING.md): 수집 → 층별 학습 → 혼합 폭 → CFG 증류 → ONNX 내보내기.
- [전체 작업 색인](WORK_LOG.md): 1~4차 기록과 성능·ASR·음향 지표.
- [공개 범위](docs/PUBLICATION.md): 제외 파일, 비식별 처리, 기록 해시 검증 범위.

`distill/corpus.json`과 `distill/corpus_cfg_v3.json`의 시험은 모두 결과 확인에 사용했습니다. 후속 모델 선택에 재사용하면 독립 시험이 아닙니다. 훈련은 고유 문장 40개를 5개 음색/seed 조합으로 늘린 200조합이며 대규모 데이터셋 학습이 아닙니다. 증류의 바이트 단위 재현은 보장하지 않습니다.

1GB 서버에서는 원본과 여러 모델을 함께 로드하는 `oracle_benchmark.sh`를 그대로 실행하지 마세요. 완성된 모델 하나를 `memory_guard.py`와 `benchmark.py`로 측정하세요. 감시기는 50ms 표본 방식의 소프트 제한이며 OS 강제 메모리 상한이 아닙니다. Docker 경로는 작성했지만 빌드·실행 검증하지 않았습니다.

## 참고와 라이선스

[Mesmer TTS](https://mesmer.tools/free-tools/tts-generator)의 실행기·증류·DSP 분리 측정과 [Paradee 8M v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1)의 작은 모델·저프레임률 헤드를 참고했습니다. [PDN 공개 번들 조사](docs/PDN_INSPECTION.json) 후 자체 C 커널을 작성했으나 서버에서 더 느려 채택하지 않았습니다. 독립 PDN 엔진, Paradee식 새 DSP 헤드, 커스텀 보코더를 구현한 결과는 아닙니다.

코드는 [MIT](LICENSE), 원본·파생 모델은 별도 [Open RAIL-M](docs/licenses/SUPERTONIC_MODEL_LICENSE) 조건을 따릅니다. [제3자 고지](THIRD_PARTY_NOTICES.md)를 참조하세요.
