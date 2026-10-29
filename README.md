# supertonic3-optimization

[Supertonic 3](https://huggingface.co/Supertone/supertonic-3)를 **CPU 한 스레드·실제 denoising 8회**로 실행하기 위한 최적화 프로젝트입니다. 현재 기준 모델은 **F1200**, Linux x86 AVX2에서 검증한 선택 엔진은 **v4**입니다.

[모델 다운로드](https://huggingface.co/babihoba/supertonic3-optimization) · [학습 데이터](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data) · [재현 안내](docs/REPRODUCING.md) · [전체 실험 기록](WORK_LOG.md)

## 현재 성능

**Oracle x86 AMD EPYC 7551 · 물리 1코어 · RAM 약 1GB**, 논리 CPU 하나에 고정하고 연산 스레드를 1로 제한했습니다. 표는 **M1 음색·KO/EN 짧은 문장**의 워밍업 후 전체 음성 생성 시간이며 모델 로딩·파일 저장은 제외합니다.

| F1200 + 엔진 v4 | 한국어 RTF | 영어 RTF | 설정 |
| --- | ---: | ---: | --- |
| 정배속 | **1.003** | **1.004** | 발화 속도 1.0× |
| 내부 배속 후보 | **1.351** | **1.354** | 내부 생성 1.35× → 출력 2.0× |

**RTF = 생성 시간 ÷ 출력 음성 길이. 작을수록 빠르며, 1 미만이면 실시간보다 빠릅니다.** RTF 1.35는 10초 음성을 생성하는 데 약 13.5초가 걸린다는 뜻입니다. 최신 검증의 표본 최대 프로세스 트리 RSS는 약 **395MB**입니다. [측정 원본](records/2026-10-29/round9_state.json)

서버를 다른 작업과 공유하므로 측정 시점에 따라 변동합니다. 과거 v2에서 RTF 0.943~0.971도 기록했지만 지속적인 실시간 생성 보장은 아닙니다. 정배속과 2배속은 출력 길이와 내부 계산량이 달라 RTF를 그대로 비교하면 안 됩니다.

## 버전별 최적화와 개선율

아래의 **RTF 감소율**은 `(1 − 개선 후 RTF / 비교 기준 RTF) × 100`입니다. 모델 변경과 엔진 변경을 구분하고 각 행에 비교 기준을 표시했습니다.

### 1. 모델 최적화

같은 EPYC 서버·같은 짧은 KO/EN 문장·실제 8회 생성의 역사적 결과입니다. 모델이 달라 출력 길이도 달라지며, 서로 다른 측정 배치의 관측값입니다.

| 단계 | 한국어 RTF | 영어 RTF | RTF 감소율: KO / EN | 바꾼 내용 |
| --- | ---: | ---: | --- | --- |
| [전체 INT8 기준](results/epyc_full_int8_8.json) | 2.220 | 2.198 | 기준 | 원래 CFG4 계산을 유지한 양자화 모델 |
| [혼합 폭 학생](results/epyc_mixed_selected_v1_8.json) | 1.879 | 1.844 | **15.4% / 16.1%** · 전체 INT8 대비 | 층별 FFN 폭을 달리 줄이고 교사 출력으로 학습, CFG4 유지 |
| [B: CFG 증류](results/epyc_cfg_selected_v1_8.json) | 1.095 | 1.106 | **41.7% / 40.0%** · 혼합 폭 대비 | CFG4의 조건·무조건 분기 결합을 단일 조건 경로가 학습 |
| [E400](results/round6_paired_E_F.json) | 1.056 | 1.069 | 속도 개선 목적의 비교 아님 | 원본 FP32·CFG4·50스텝 교사로 8스텝 학생 재학습, 400회 |
| [F1200 · 현재 기준](results/round6_paired_E_F.json) | 1.052 | 1.070 | E400 대비 **일관된 속도 개선 확인 못함** | 같은 원본 교사로 1,200회 학습, 청취 후 기준 모델로 선택 |

B까지의 관측 RTF는 전체 INT8 기준보다 KO **50.7%**, EN **49.7%** 낮았습니다. 전체 INT8 기준도 이미 최적화된 모델입니다. 원본 FP32는 1GB 서버의 메모리 감시에서 중단됐으므로 **원본 대비 서버 가속률은 제시하지 않습니다**.

폭 1024 후보는 B 대비 RTF가 약 **19.2% / 19.5%** 더 낮았지만 발음 오류와 청취 품질 저하로 채택하지 않았습니다. 현재는 F1200의 크기와 가중치를 고정하고 엔진을 개선합니다. [모델·품질 실험 기록](records/2026-10-29/ROUND5_LOCAL.md)

### 2. 엔진·배속 최적화 — F1200 가중치 고정

| 버전 | 한국어 / 영어 RTF | 비교 기준 | RTF 감소율: KO / EN | 최적화 방법 |
| --- | --- | --- | --- | --- |
| [엔진 v1](results/round6_paired_final.json) | 1.034 / 1.032 | 기본 F1200 · 속도 1.05× | **1.6% / 1.6%** | 정확한 AVX2 정수 행렬곱, 가중치 packing 재사용, 문장 조건 캐시 |
| [엔진 v2](results/round6_paired_v2.json) | 0.948 / 0.957 | 기본 F1200 · 속도 1.05× | **12.1% / 9.8%** | 안전한 K8 정수 부분합의 변환 연산을 공유, 나머지는 정확한 S32 경로 사용 |
| [내부 배속 후보](records/2026-10-29/round8_state.json) | 1.414 / 1.394 | 내부 1.05× → 출력 2× 후보 · 엔진 v2 | **16.1% / 17.0%** | 내부 생성을 1.35×로 조절하고 보코더 안에서 세부 latent 프레임을 2× 길이로 압축 |
| [엔진 v4 · 현재 선택](results/round9_paired_tile_speed2.json) | 1.351 / 1.354 | v2 + 동일한 내부 배속 후보 | **4.0% / 4.5%** | K8 안전 조건을 유지한 4행 × 16열 타일, 활성값 읽기를 두 출력 타일에서 재사용 |

v1·v2·v4는 순서를 바꿔 실행한 **쌍별 RTF 감소율의 중앙값**입니다. RTF 열은 후보 실행들의 중앙값이며, 내부 배속 행은 별도 배치의 중앙값 비교입니다. **v2의 비교 기준은 v1이 아닌 기본 F1200**이며, 서로 다른 행의 감소율을 단순히 더하거나 누적 가속률로 계산하지 않습니다.

엔진 변경은 양자화 값·학습 가중치·8회 계산을 유지합니다. v1/v2는 기본 F1200과 각각 6개, v4는 기존 배속 후보와 14개 비교에서 **float32 파형이 완전히 같았습니다**. v3의 K16 부분합과 epilogue 융합은 일관된 속도 이득이 없어 채택하지 않았습니다. [엔진 v1/v2 상세](records/2026-10-29/ROUND6_ENGINE.md) · [v3 실험](records/2026-10-29/ROUND7_SPEED.md) · [v4 상세](records/2026-10-29/ROUND9_ENGINE.md)

### 3. 초기 FP32 그래프 최적화 — M4 Max 별도 측정

1×1 Conv를 행렬곱으로 바꾸고, 매 스텝 반복되던 문장 조건을 한 번만 계산했습니다. 원래 CFG4·FP32·8스텝을 유지한 CPU 한 스레드 비교에서 한국어 RTF **0.305 → 0.067**, 영어 **0.317 → 0.069**, 각각 **78.0% / 78.1% 감소**했습니다. 짧고 긴 4문장의 가속비는 4.54~4.71배였습니다. 이 결과를 EPYC·Oracle ARM 성능으로 환산하지 않습니다. [측정 원본](results/paired_fp32_cached_8.json)

## Oracle ARM 측정

사용자가 별도로 측정한 **Neoverse-N1 · 2코어 · 12GB**의 결과입니다. 대상은 **이전 B 모델**이며, F1200·엔진 v4·내부 2배속 후보의 결과와 구분합니다.

| 실행 설정 | 짧고 긴 KO/EN 4문장 RTF 범위 |
| --- | ---: |
| 한 스레드 | **0.276~0.283** |
| 두 스레드 | **0.153~0.161** |

각 문장 워밍업 후 5회 중앙값이며, 두 설정의 저장 음성 4쌍은 PCM16 샘플이 같았습니다. 기본 실행 설정은 한 스레드입니다. v4 native 커널은 Linux x86_64 AVX2용이며 ARM은 표준 ORT 경로를 사용합니다. [ARM 측정 PR #1](https://github.com/8954sood/supertonic3-optimization/pull/1)

## 빠른 시작

Python 3.12 이상을 사용합니다. Git에는 소스와 기록을, Hugging Face에는 모델·체크포인트·데이터를 보관합니다. 기본 F1200 패키지는 약 **194MB**이며 공개 다운로드에 토큰이 필요하지 않습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_artifacts.py --purpose inference
python synthesize.py '안녕하세요. 하나의 스레드로 음성을 만듭니다.' \
  --model .hf-assets/model/models/round5-original50-student \
  --lang ko --steps 8 --speed 1.0 --output samples/hello.wav
```

위 명령은 이식 가능한 표준 ORT 패키지를 실행합니다. **v4 커널 빌드·그래프 변환·제한 서버 측정은 [EPYC 엔진 설정](docs/REPRODUCING.md#epyc용-엔진-실험)**을 따라 별도로 준비하세요. 변환과 학습은 RAM이 넉넉한 개발 환경에서 수행합니다.

### 내부 2배속 후보 사용

```bash
python download_artifacts.py --purpose inference --model round8-latent-retime-g135
python synthesize.py '안녕하세요. 오래된 컴퓨터에서도 빠르게 음성을 생성합니다.' \
  --model .hf-assets/model/models/round8-latent-retime-g135 \
  --lang ko --voice M1 --steps 8 --speed 2.0 --output samples/fast.wav
```

F1200 가중치와 8회를 유지하며 **추가 학습·추가 파라미터·파형 생성 후 배속 필터 없이** 보코더 그래프 안에서 처리합니다. 2.0×에만 새 경로를 적용하고 다른 속도는 기존 경로를 사용합니다.

기본 duration 방식의 2배속에서 확인한 누락·뭉개짐을 줄였고, 짧은 남성 샘플은 사용자 청취에서 개선됐습니다. 다만 확대 개발 검증 16개에서 **한국어 CER 51/193(26.42%), 영어 WER 1/96(1.04%)**이 남습니다. 모든 문장의 누락 해결이나 원본 음질 동등성을 보장하지 않으며, ASR 오류율은 지각 음질 점수가 아닙니다. [배속 품질·실패 시도·검증 범위](records/2026-10-29/ROUND8_NATIVE_SPEED.md)

## 서버·스트리밍

원본 형식의 `/v1/tts`, `/v1/audio/speech`, 배치·음색 API와 **문장 묶음 단위 PCM·SSE 스트리밍**을 제공합니다. 모델을 한 번 로드해 재사용하며 묶음마다 8스텝을 유지합니다.

```bash
python -m pip install -r requirements-serve.txt
python serve.py --model .hf-assets/model/models/round5-original50-student
```

브라우저에서 <http://127.0.0.1:7788/>로 접속하면 생성 중인 음성을 바로 들을 수 있습니다. 2배속 후보는 `--model`에 다운로드한 `round8-latent-retime-g135` 경로를 지정합니다. [서버 API·스트리밍 형식·첫 오디오 지연·측정 안내](docs/SERVER.md)

## 재현과 기록

다운로드 도구는 고정 Hugging Face revision과 파일별 SHA-256을 검사합니다. F1200 초기·최종 체크포인트, E400 비교 모델, 원본 50스텝 교사 궤적 **640개 학습 쌍·128개 검증 쌍**을 공개했습니다. 교사는 원본 FP32·CFG4·50스텝이며, 기존 학생은 초기값으로만 사용했습니다.

```bash
python download_artifacts.py --purpose reproduce
```

| 자료 | 내용 |
| --- | --- |
| [재현 안내](docs/REPRODUCING.md) | 다운로드, 학습·내보내기, v4 준비, 성능·품질 측정 |
| [F1200 공개 검증](records/2026-10-29/PUBLICATION_F1200.md) | 모델·데이터 revision, 파일 해시, 재현 체크포인트 |
| [내부 배속 패키지 공개 검증](records/2026-10-29/PUBLICATION_RETIME.md) | 공개 다운로드와 출력 일치 확인 |
| [전체 작업 색인](WORK_LOG.md) | 버전별 시도, 미채택 후보, 원시 측정, 후속 과제 |

기존 시험 문장들은 결과 확인에 사용했습니다. 후속 모델 선택을 위한 독립 시험은 새로 고정해야 하며, 재학습 결과의 바이트 단위 일치는 보장하지 않습니다.

## 참고와 라이선스

[Mesmer TTS](https://mesmer.tools/free-tools/tts-generator)의 실행 구조와 [Paradee 8M v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1)의 경량 모델·저프레임률 헤드를 참고했습니다. PDN 엔진이나 Paradee 보코더를 이식한 구현은 아닙니다. 코드는 [MIT](LICENSE), 원본·파생 모델은 [Open RAIL-M](docs/licenses/SUPERTONIC_MODEL_LICENSE)을 따릅니다. [제3자 고지](THIRD_PARTY_NOTICES.md)
