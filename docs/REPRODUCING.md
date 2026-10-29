# 학습·검증 재현

## Hugging Face에서 모델과 학습 자료 받기

공개 [모델 저장소](https://huggingface.co/babihoba/supertonic3-optimization)에는 세 개의 완전한 ONNX 모델과 `mixed-v1-initial.pt`, 선택된 `cfg-mixed-v1-step-00600.pt`가 있습니다. [데이터셋 저장소](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data)에는 1,600개 훈련 쌍과 64개 검증 쌍이 있습니다. 파일별 SHA-256은 각 저장소의 `asset_manifest.json`에서 확인합니다. 전체 저장소는 약 1.2GB입니다. 아래 도구는 선택 모델만 약188MB, 재현 필수 자료만 약646MB를 내려받습니다. 모델 revision은 `85bc100ca8d7cefad17b63a4c7ba489638dbe2f2`, 데이터셋 revision은 `817056ae3e9b8fa1a2e464a212bb2ba1f4c5196a`로 고정하고 파일별 크기·SHA-256을 확인합니다.

```bash
python download_artifacts.py --purpose inference
python download_artifacts.py --purpose reproduce
```

추론할 선택 모델:

```bash
python synthesize.py '안녕하세요.' --model .hf-assets/model/models/cfg-mixed-v1-selected \
  --lang ko --steps 8 --output samples/hello.wav
```

체크포인트와 교사 궤적 데이터로 선택 모델의 증류 단계를 재현합니다. 다음 받은 파일을 그대로 지정하면 됩니다:

```text
.hf-assets/model/reproduction/checkpoints/mixed-v1-initial.pt
.hf-assets/data/trajectories-v2
```

첫 번째 파일은 600회 CFG 증류의 초기 혼합 폭 가중치이고 두 번째 경로는 훈련/검증 궤적입니다. 원본 Supertonic 모델은 아래 설치 단계의 `download.py`로 고정 리비전에서 받습니다. 새 학습은 하드웨어와 라이브러리 버전에 따라 가중치가 달라질 수 있습니다.

프로젝트를 설치한 뒤 원본 Supertonic을 받고 export용 FP32 캐시 패키지를 만듭니다. 모델·체크포인트는 Git에 포함되지 않으며 위 Hugging Face 저장소에서 받습니다. 학습에는 여러 GB의 RAM·디스크와 Mac MPS가 필요합니다. CPU1·1GB 조건은 완성된 모델의 추론 측정 조건입니다.

## 혼합 폭 CFG 증류 재실행

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
