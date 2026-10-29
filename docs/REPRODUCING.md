# F1200 다운로드·재현·엔진 측정

현재 사용자가 선택한 모델은 **F1200** (`round5-original50-student`)입니다. 구조와 가중치를 고정하고 엔진만 최적화합니다. E400은 청취 비교용입니다. 새 teacher target은 pinned 원본 FP32·CFG4·50스텝으로만 만들며 학생은 실제8회 추론합니다.

[모델·체크포인트](https://huggingface.co/babihoba/supertonic3-optimization/tree/d33f1094c6057d8c613f2c45987ba45e57c4b3af) revision은 `d33f1094c6057d8c613f2c45987ba45e57c4b3af`, [교사 데이터](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data/tree/e56d93c9959647c850b639a614e629e4e4ac1593)는 `e56d93c9959647c850b639a614e629e4e4ac1593`입니다. 파일 크기·SHA256을 확인하고 공개 다운로드에 토큰을 사용하지 않습니다.

## 추론만 실행

2배속 완성 패키지를 추가한 현재 모델 다운로드 revision은 **`f1d05bdb691d8769557e19644f3a63089992daf0`**입니다. 이전 revision의 기존118개 파일은 변경하지 않았으며 교사 데이터 revision도 그대로입니다. 기본 선택은 F1200을 유지합니다. [선택2배속 패키지](https://huggingface.co/babihoba/supertonic3-optimization/tree/f1d05bdb691d8769557e19644f3a63089992daf0/models/round8-latent-retime-g135), [공개 검증](../records/2026-10-29/PUBLICATION_RETIME.md).

```bash
python -m pip install -r requirements.txt
python download_artifacts.py --purpose inference
python synthesize.py '안녕하세요. 한 스레드로 읽습니다.' \
  --model .hf-assets/model/models/round5-original50-student \
  --lang ko --steps 8 --output samples/hello.wav
```

약194MB의 완전한 F1200 ONNX 패키지에 설정·10음색·모델 라이선스가 포함됩니다. 모델/체크포인트/음성은 Git에 넣지 않습니다. `--model round5-original50-preview`로 E400만 받을 수도 있습니다. 이전 모델 이름도 명시적으로 지정할 수 있습니다.

## 말하기 배속

`synthesize.py`와`benchmark.py`의 `--speed`로0.7~2.0을 지정합니다. 기본1.05를 유지하며1.0이 기준 속도입니다. 1.5는 발화 길이를 약2/3,2.0은 약1/2로 줄입니다. 오디오 재생 샘플레이트를 바꾸는 방식이 아니라 원래 모델의duration 조건을 조절합니다. 범위 밖·NaN·무한대는 거부합니다. **현재 native2.0에서는 원본50과 F1200 모두 누락·발음 문제가 확인됐습니다.** 1.5도 음색에 따라 오류가 났습니다. 입력 가능 범위는 품질 보장이 아닙니다.

```bash
python synthesize.py '안녕하세요. 배속을 확인합니다.' \
  --model .hf-assets/model/models/round5-original50-student \
  --lang ko --voice M1 --steps 8 --speed 1.5 --output samples/fast.wav
```

배속별 길이·float32 출력·개발ASR 검사 범위와 한계는 [7차 기록](../records/2026-10-29/ROUND7_SPEED.md)을 참고하세요. 빠른 발화의 지각 품질은 길이 비례만으로 검증할 수 없습니다.

### 모델 내부2배속 후보 재현

완성된 패키지는 다음처럼 받을 수 있습니다. 새 훈련 checkpoint나 데이터셋은 필요하지 않습니다.

```bash
python download_artifacts.py --purpose inference --model round8-latent-retime-g135
python synthesize.py '안녕하세요. 오래된 컴퓨터에서도 빠르게 음성을 생성합니다.' \
  --model .hf-assets/model/models/round8-latent-retime-g135 \
  --lang ko --steps 8 --speed 2 --output samples/retime.wav
```

아래는 기존 F1200에서 동일 그래프를 직접 준비하는 방법입니다.

```bash
python -m pip install -r requirements-train.txt
python prepare_latent_retime.py \
  --base .hf-assets/model/models/round5-original50-student \
  --output models/F1200-retime --generation-speed 1.35
python experiments/check_latent_retime.py \
  --base .hf-assets/model/models/round5-original50-student \
  --candidate models/F1200-retime --output results/retime-regression.json
python synthesize.py '안녕하세요. 오래된 컴퓨터에서도 빠르게 음성을 생성합니다.' \
  --model models/F1200-retime --lang ko --steps 8 --speed 2 --output samples/retime.wav
```

이 준비는 공개 F1200만 필요하고 재학습·새 체크포인트 다운로드가 없습니다.144채널을24채널×6프레임으로 풀어 세부 시간축을 압축한 후 기존 보코더로 파형을 만듭니다. 정상1.05 출력4개는 원래 F1200과 완전히 일치했고,2.0에서도 실제 vector 호출8회를 검사했습니다. 실제 생성 길이는 내부1.35이므로 native2의 짧은 길이8회보다 계산량이 많습니다. 서버 RTF와16개 검증에 남은 오류는 [8차 기록](../records/2026-10-29/ROUND8_NATIVE_SPEED.md)을 참고하세요.

EPYC용 조합은 위 패키지를 `experiments/exact_integer_backend.py --source models/F1200-retime --output models/F1200-retime-exact --cache-conditioning`에 넣으면 됩니다. 먼저 아래 엔진 절차대로 native 커널을 빌드해야 합니다. ARM은 표준 ORT 패키지를 사용하고 별도로 측정하세요. 새 패키지는2.0만 내부 압축하며 나머지 배속은 기존 경로입니다. 추론에 FFmpeg/Rubber Band가 필요하지 않습니다.

## 원본50 교사에서 F1200 단계 재현

```bash
python -m pip install -r requirements-train.txt
python download_artifacts.py --purpose reproduce
python download.py
python prepare.py --mode fp32-cached --output models/fast-fp32-cached
python experiments/guidance.py conditional --source models/supertonic-3 \
  --output models/cfg-student-fp32-source
python distill/train_joint.py \
  --checkpoints .hf-assets/model/reproduction/round5/original50-initial.pt \
  --data .hf-assets/data/round5/trajectories-original50-round5 \
  --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx --train-all \
  --iterations 1200 --learning-rate 1e-5 --validate-every 200 \
  --output distill/original50-reproduced
python distill/export_joint.py \
  --checkpoint distill/original50-reproduced/best.pt \
  --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx \
  --base models/fast-fp32-cached --int8 --cfg-distilled \
  --output models/F1200-reproduced
```

재현 다운로드는 초기/최종 체크포인트 약339MB와 데이터 archive 약143MB 및 작은 보고서만 받습니다. archive의769개 멤버(640train+128validation NPZ, manifest)를 안전하게 풀고 개별 SHA256까지 확인합니다. 고유 문장은80train/16validation, validation update loss는 선택한48쌍으로 계산했습니다. 원본50 궤적을8스텝 시점에 보간하고 마지막은 원본 최종latent를 사용했습니다. 새 독립 시험은 평가하지 않았습니다.

초기 checkpoint를 제공하므로 과거 층별 FFN fit과 B 초기화 과정을 다시 돌릴 필요가 없습니다. 기존 학생은 초기값으로만 사용했고, 새로운 학습 목표로 쓰지 않았습니다. 학습에는 여러GB RAM이 필요합니다. 1코어·1GB는 완성된 모델의 서버 추론 조건입니다. 하드웨어/라이브러리가 바뀐 재학습이 바이트 단위로 같다고 보장하지 않습니다. 받은 최종 checkpoint로 export만 하면 배포 ONNX와 대조할 수 있습니다.

```bash
python distill/export_joint.py \
  --checkpoint .hf-assets/model/reproduction/round5/original50-step-01200.pt \
  --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx \
  --base models/fast-fp32-cached --int8 --cfg-distilled \
  --output models/F1200-reexported
```

교사 데이터를 다시 만들려면 `distill/collect_original_multistep.py`와 `experiments/audit_original_trajectories.py`를 사용합니다. 원본 teacher revision과 네 graph SHA256은 데이터 manifest에 있습니다. 파생 교사 데이터는 `distill/teacher_policy.py`가 거부합니다. 이미 존재하는 출력 경로를 덮어쓰지 말고 새 이름으로 실행하세요.

## EPYC용 엔진 실험

기본 F1200 패키지는 표준 ORT 연산으로 실행됩니다. 추가 native 커널은 **Linux x86_64 AVX2·EPYC7551에서만 검증**합니다. ARM용으로 제공하는 커널이 아니며 ARM에서는 우선 기본 F1200을 측정하세요. 모델 파라미터와 양자화는 바꾸지 않습니다.

```bash
python -m pip install ziglang==0.17.0
python native/build_exact.py --cpu znver1
python experiments/exact_integer_backend.py \
  --source .hf-assets/model/models/round5-original50-student \
  --output models/F1200-exact-cached --cache-conditioning
python experiments/check_exact_integer.py
python memory_guard.py --max-rss-mb 480 --reserve-mb 80 --seconds 600 \
  --stats results/F1200-engine-guard.json -- \
  python benchmark.py --model models/F1200-exact-cached --steps 8 \
  --repeats 3 --cases ko_short en_short ko_long en_long --affinity 0 \
  --audio-subtype FLOAT --output results/F1200-engine.json
```

커널은 상수 가중치를 한 번 packing하고 U8×S8 정수 결과를 정확히 계산합니다. 문장 조건을1회 계산하며 길이가 다른 문장의 memory pattern 누적을 끕니다. CPU arena와 ORT prepacking은 유지합니다. 기본 모델을 먼저 같은 조건으로 측정하고 WAV/JSON을 비교하세요. 감시기는50ms 표본이며 하드 메모리 제한이 아닙니다. [6차 기록](../records/2026-10-29/ROUND6_ENGINE.md)에 실패한 메모리 설정과 재측정도 남깁니다.

현재 `native/build_exact.py`는 엔진v4를 빌드합니다.4행×16열 tile에서도 기존 K8 안전 경계와 S32 fallback을 유지하며 학습 가중치·quantization·8회 계산은 바꾸지 않습니다. 그래프 변환이나 모델 다운로드를 다시 하지 않고 기존 exact 패키지의 라이브러리만 재빌드할 수도 있습니다. 내부2배속 조합도 위에서 준비한 `models/F1200-retime`을 exact 변환의 source로 지정하면 됩니다. 별도 epilogue fusion은 일관된 이득이 없어 미채택이며 기본 그래프에 넣지 않습니다.

9차 실제 서버에서2배속 짧은KO/EN의 쌍별 가속비 중앙값은1.042/1.047배이고 RTF는1.351/1.354였습니다. 전체 파형 보존 검사와 정상·장문 조건은 [9차 기록](../records/2026-10-29/ROUND9_ENGINE.md)에 있습니다. ARM 가속 커널로 해석하지 마세요.

로컬 정리로 오래된 실험 가중치 대부분을 삭제했으므로 과거 기록의 경로에 manifest만 남아 있을 수 있습니다. 보존한 F1200·원본 교사·선택 retime 후보·초기/최종 checkpoint 외의 가중치가 필요하면 고정된 공개 HF revision에서 **새 폴더**로 받거나 해당 실험 코드를 재실행하세요. 기존 데이터·결과 기록·청취 파일은 유지했습니다.

## 과거 B 모델 재현

아래는 Hub의 초기 checkpoint와 궤적 데이터에서 선택 후보를 다시 학습·내보내는 최소 경로입니다. 저장된 모델을 단순히 사용하려면 재학습 없이 위의 `synthesize.py` 명령을 실행하세요.

```bash
python -m pip install -r requirements-train.txt
python download.py
python prepare.py --mode fp32-cached --output models/fast-fp32-cached
python experiments/guidance.py conditional --source models/supertonic-3 \
  --output models/cfg-student-fp32-source
python distill/train_joint.py \
  --checkpoints .hf-assets/model/reproduction/checkpoints/mixed-v1-initial.pt \
  --data .hf-assets/data/trajectories-v2 \
  --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx --train-all \
  --iterations 2400 --learning-rate 3e-5 --validate-every 200 \
  --output distill/joint-cfg-mixed-v1-retrained
python distill/export_joint.py \
  --checkpoint distill/joint-cfg-mixed-v1-retrained/best_step_00600.pt \
  --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx \
  --base models/fast-fp32-cached --int8 --cfg-distilled \
  --output models/cfg-mixed-v1-retrained
```

Hub에는 이 단계 전에 학습한 혼합 폭 초기 가중치도 함께 저장했습니다. 그래서 위 과정에 별도의 2.2GB 층별 activation 수집과 512/1024/1536 폭 사전학습은 필요하지 않습니다. 전체 초기화부터 다시 학습하려면 Git의 `distill/collect.py`, `distill/train.py`, `distill/init_mixed.py` 코드를 사용하고 새 설정·결과를 별도 기록하세요.

600회 체크포인트는 기존 선택 후보의 출발점입니다. 새 훈련에서 해당 시점이 best로 저장되지 않거나 하드웨어·연산 차이가 발생하면 기존 가중치와 같지 않습니다. 현재 스크립트는 **검증 손실 best 갱신 시점**을 별도 파일로 남깁니다. 새 실행의 검증과 청취로 선택하고 기존 결과를 그대로 붙이지 마세요. `docs/models/`의 SHA-256은 과거 선택 ONNX의 식별자이며 새 훈련 출력의 일치를 보장하지 않습니다.

층별 원본 FFN은 2048 폭이고 혼합 폭은 512×2, 1024×4, 1536×12, 2048×10입니다. 반복부 파라미터는 64,014,576→50,369,776입니다. frontend/decoder는 FP32, 반복부는 감소 범위·채널별 동적 INT8입니다. 내보내기에서 기존 `condition.onnx`를 제거하고 완전한 단일 경로 그래프를 저장합니다.

## 평가

```bash
python -m pip install -r requirements-eval.txt -r requirements-acoustic.txt
python distill/validate_graphs.py --data .hf-assets/data/trajectories-v2 --output results/retrained_graph_parity.json
python distill/evaluate.py --variants models/cfg-mixed-v1-retrained \
  --split validation --output results/retrained_validation.json
python experiments/acoustic_metrics.py results/retrained_validation.json \
  --output results/retrained_acoustic.json
```

음향 분석은 생성한 WAV가 필요하며 WAV는 Git에 없습니다. 원시 JSON에는 당시의 상대 WAV 경로를 보존했습니다. Whisper base ASR은 별도 다운로드하며 KO는 공백 제거 문자 오류율, EN은 단어 오류율입니다. logmel/pYIN/DTW는 대리 지표이며 MOS·화자 유사도·음질 일치율이 아닙니다. 커스텀 harmonic/interharmonic 지표는 Paradee의 수치와 직접 비교할 수 없습니다.

기존 두 test 세트는 이미 소비했습니다. 위의 새 모델 검증을 위해 기존 test 점수를 보고 학습하거나 선택하지 말고, 독립 시험은 새로 작성·고정하세요. 고유 train 문장은 40개뿐입니다.

## 제한된 서버에서 측정

완성된 모델 하나와 추론 파일만 복사하고 다음을 실행합니다. 설치·모델 변환·ASR 학습을 제한 서버에서 동시에 돌리지 마세요.

```bash
python memory_guard.py --max-rss-mb 420 --reserve-mb 80 --seconds 300 \
  --stats results/guard_selected.json -- \
  python benchmark.py --model models/cfg-mixed-v1-retrained --steps 8 \
    --repeats 3 --cases ko_short en_short --affinity 0 \
    --output results/server_selected.json
```

`--affinity`에는 해당 프로세스에 허용된 논리 CPU를 지정합니다. 감시기는 Linux cgroup의 강제 상한이 아닙니다. ARM과 x86 각각에서 모델 해시와 출력·속도를 다시 확인해야 합니다. 구형 AVX2에서 양자화 출력 차이가 컸으며 reduced-range만으로 전부 해결되지 않았습니다.

현재 선택 모델을 별도로 보유한 경우 다음으로 최소 실행 패키지를 만들 수 있습니다. 패키지는 Git에서 제외됩니다.

```bash
python package_runtime.py --model models/cfg-mixed-v1-selected \
  --output dist/cfg-mixed-v1-runtime.tar.gz
```

재현 확인: 공개 체크포인트로 내보낸 네 ONNX 파일의 SHA-256이 기존 배포본과 전부 일치했습니다. 데이터 manifest의 이전 작업 경로는 현재 `--data` 경로로 재해석하므로 다운로드 위치를 옮길 수 있습니다.
