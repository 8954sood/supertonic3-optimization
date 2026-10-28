> 당시 시점의 기록입니다. 최신 확정 상태는 [HANDOFF_ROUND3.md](HANDOFF_ROUND3.md)를 참조하세요.

# 다음 작업 인계

이 문서의 실행 상태는 **1차 스냅샷**이다. 이후 작업은 [최신 인계](HANDOFF_ROUND2.md)와 [2차 결과](ROUND2.md)를 먼저 읽는다.

## 먼저 읽을 것

1. 이 폴더의 `README.md` 전체: 결과와 실패, 음질 한계.
2. `metrics.json`: 성능/CER/WER/파형/학습 수치. 원문은 각 행의 `source`.
3. `distill/README.md`와 `distill/corpus.json`: 학습 방식과 데이터 분리.
4. 프로젝트 `README.md`: 설치·기본 실행. 일부 초기 문서보다 이 날짜의 기록이 최신 판단이다.

현재 위치는 `<WORKSPACE>`. 사용자가 원본 대비 어색함을 직접 지적했고, **음질 개선이 미완료**다. ASR 통과를 품질 통과로 취급하지 않는다. 사용자는 증류·재학습을 허용했고 계속 시도하길 원한다. 가장 최근 요청은 새 실험보다 기존 작업을 먼저 기록하라는 것이다.

## 종료 시점 실행 상태

기록 시점까지 시작한 학습은 모두 끝났다. `distill/student1024`, `student1536`, `student512`, `student256` 각각 28층 완료. `pilot1024`는 1층 파일럿이다. 폭512 검증과 서버1024 벤치마크도 끝났다. 백그라운드 학습이 계속된다고 가정하지 말고 실행 여부를 먼저 확인한다. 기록 작성 중 새 추론·학습은 시작하지 않았다.

- 1024 FP32/INT8/QAT: 모델 존재. INT8만 서버 실측. 여러 음색 검증 CER/WER는 원본과 동일하지만 원본 ASR 문자 편집 일치는 KO98.86%. 사용자 청취 문제 때문에 보류.
- 1536 INT8: 모델 내보내기 완료. ASR/음향/서버 결과 없음.
- 512 INT8/QAT: 검증 발음 악화. 현재 설정 제외. 서버 미실행.
- 256: 체크포인트만 완료, ONNX 내보내기 없음. 품질·속도 미측정.
- 별도 `test` 12문장은 **아직 사용하지 않았다**. 후보 선택은 validation만 쓸 것.

## 주요 파일

| 위치 | 역할 |
| --- | --- |
| `runtime.py` | CPU1, 8~50스텝 검증, 원본/부분모델 로딩, 고정 seed 추론 |
| `benchmark.py` | 워밍업·반복·단계별 시간·CPU/wall·RSS·WAV |
| `memory_guard.py` | 서버에서 본 작업의 자식만 RSS/가용 RAM/시간 감시 |
| `prepare.py`, `optimize.py` | 완전 패키지와 조건 캐시/INT8 |
| `experiments/pointwise.py` | 1×1 Conv→MatMul, 동적 INT8 |
| `experiments/guidance*.py` | 중복 제거와 가이던스 변경(후자는 품질 손실 실험) |
| `native/`, `native_ops.py` | 자체 C 연산. 서버에서 Q8보다 느림 |
| `distill/collect.py` | 원본 ONNX 교사의 층별 입출력 수집 |
| `distill/train.py` | 선택 초기화·ridge·AdamW·QAT, 체크포인트 |
| `distill/export.py` | 학생 FFN을 원본 ONNX에 넣고 INT8 변환 |
| `distill/evaluate.py` | validation/test 다중 음색 paired ASR |
| `experiments/asr_check.py` | 재사용된 개발용 6문장 검사 |
| `experiments/evaluate_remote_audio.py` | 서버가 실제 만든 WAV만 ASR |
| `experiments/build_record.py` | 기존 결과 JSON과 모델 목록 취합; 추론 안 함 |

## 재현 명령

현재 `.venv`를 사용한다. 기본 Python>=3.12, `requirements.txt`; ASR는 `requirements-eval.txt`, 훈련은 `requirements-train.txt`. 새 설치는 원본 모델의 고정 리비전을 받도록 `download.py`를 쓴다. Mac GPU는 훈련에만 사용했다.

```bash
.venv/bin/python synthesize.py '안녕하세요. 원본을 기준으로 음질을 비교합니다.' --model models/supertonic-3 --steps 8 --output samples/reference.wav
.venv/bin/python synthesize.py '안녕하세요. 원본을 기준으로 음질을 비교합니다.' --model models/fast-fp32-cached --steps 8 --output samples/fp32_cached.wav
.venv/bin/python synthesize.py '안녕하세요. 원본을 기준으로 음질을 비교합니다.' --model models/student1024-int8 --steps 8 --output samples/student1024.wav
```

`--model`은 패키지 루트, `--variant`는 해당 패키지의 **onnx 폴더**를 받는다. `distill/evaluate.py --variants`는 패키지 루트 목록을 받는다. 두 방식 혼동 주의.

```bash
.venv/bin/python distill/evaluate.py --variants models/student1024-int8 --split validation --output results/new_validation.json
.venv/bin/python distill/train.py --output distill/student1024 --width 1024 --iterations 1600 --qat-steps 600
```

학습 스크립트는 완료된 동일 폭·반복수 층을 건너뛰며 미완료 층은 재시작한다. 현재 skip 조건은 폭/iterations만 비교하므로 learning rate·데이터·QAT 설정을 바꾸면 **새 출력 폴더**를 사용해야 한다. 저장된 optimizer가 있어도 미완료 층의 optimizer 재개는 아직 구현하지 않았다. 기존 출력 모델은 export가 덮어쓰지 않는다.

현재 정해진 teacher corpus 수집을 다시 실행하지 말고 `distill/data/manifest.json`과 `train/`, `validation/`의 mmap을 재사용한다(약2.2GB, 실험 데이터). `report.json`, `layer_XX.pt`, `results/train*.log`를 보존한다. QAT 체크포인트와 float 체크포인트를 혼동하지 않는다. 최종 INT8 내보내기 자체도 손실 변환이다.

## 서버 재현과 안전한 범위

접속 계정/호스트는 사용자가 제공한 `<SSH_USER>@<SSH_HOST>`, 개인키 파일은 작업 폴더의 `<SSH_KEY_FILE>`이다. **개인키 내용을 열거나 출력·복사·아카이브하지 말 것.** 사용자 승인은 기존 세션에 있으므로 일반적인 본 작업 재측정에 새 승인 요청은 필요 없다.

서버 작업 폴더: `<REMOTE_WORKSPACE>`. 그 안의 `.venv`와 원본/여러 실험 모델을 사용한다. 시스템 서비스나 다른 프로그램을 건드리지 않는다. cgroup memory controller가 없는 커널이다. `oracle_benchmark.sh`는 12GB ARM용이며 원본/후보 동시 로딩도 하므로 **이 1GB 서버에서 그대로 실행하지 않는다**.

```bash
ssh -i <SSH_KEY_FILE> -o BatchMode=yes <SSH_USER>@<SSH_HOST>
cd <REMOTE_WORKSPACE>
nice -n 19 .venv/bin/python memory_guard.py --max-rss-mb 420 --reserve-mb 80 --seconds 240 --stats results/guard_NEW.json -- .venv/bin/python benchmark.py --model models/student1024-int8 --steps 8 --repeats 3 --cases ko_short en_short --affinity 0 --output results/epyc_NEW_8.json
```

새 이름으로 결과를 저장한다. 계측용 memory_guard는 하드 제한이 아니며 순간 피크를 놓칠 수 있다. Mac의 `libpacked_blas.so`는 x86_64 Linux용 교차 빌드다. ARM에 그대로 쓰면 안 된다. ORT1.20.1은 `ort120/` 실험 폴더에만 있고 기본 .venv는1.31.1이다.

## 음질 개선을 재개할 순서

1. **무엇이 어색한지 분리**: 동일 text/voice/seed/speed/8스텝에서 원본FP32, FP32캐시, 전체INT8, 반복부만INT8, 학생FP32, 학생INT8, CFG변경 후보를 비교한다. 현재 사용자 청취 파일명은 특정되지 않았다. 재생용 A/B 파일을 명확히 구분해 제공한다.
2. 원본 길이 예측/텍스트 인코더/보코더를 고정한 부분 교체부터 비교한다. 전체INT8은 길이까지 바꿀 수 있으므로 단순 파형 상관으로 학생 품질을 판단하지 않는다.
3. **새 음향 지표**: 정렬된 다중해상도 STFT/log-mel 거리, 유성구간 F0 오차·상관과 유무성 일치, 음량/지속시간, 화자 유사도, 사용자가 언급한2~4kHz 배음/중간대역 에너지 등을 추가한다. 각 지표의 정의·정렬 방식·단위·실패 조건을 기록한다. 하나의 “일치율”로 합치지 않는다.
4. 1536 또는 검증 오차가 큰 층을 원본 폭으로 두는 혼합 폭을 비교할 수 있다. 512/256을 속도만으로 채택하지 않는다. 층별 입력이 교사에서 학생으로 바뀌는 분포 이동을 줄이는 student-trajectory 데이터 수집/교사 재라벨과 전체 denoiser 증류를 검토한다.
5. 더 많은 KO/EN 텍스트와 음색/길이/seed를 균형 있게 수집한다. 현재40문장은 Paradee의12,000문장 규모와 다르다. 기존 validation으로 tuning하고 test12문장은 최종 후보 고정 후 한 번 사용한다.
6. Paradee의 저프레임률 harmonic/noise 헤드는 Supertonic 보코더에 그대로 꽂을 수 없다. 별도 latent→audio 학생 디코더, 피치/유무성 조건 또는 예측기, 다중해상도 스펙트럼·배음·위상/잡음 손실을 설계하고 학습해야 한다. 적대적 훈련은 안정적인 스펙트럼 재현 후 비교한다. 아직 어느 것도 구현/검증하지 않았다고 명시한다.
7. 음질이 나아진 후보만 실제 서버에서 CPU1·8스텝 재측정한다. 속도/품질 Pareto 표를 업데이트하고 청취 피드백을 함께 기록한다. ARM 실측은 별도다.

## 기록을 갱신할 때

실험마다 목적/가설, 소스·모델·설정·데이터 split, 실제 명령, 원시 JSON/WAV/체크포인트 위치, 성능·품질 수치, 채택/보류/거절 이유를 적는다. 원시 결과를 덮어쓰지 말고 새 이름을 쓴다. 실패도 남긴다. 향후 스냅샷은 새 날짜/차수 폴더에 만들고 이번 기록을 유지한다.

`source_index.json`의 SHA-256은 원시 JSON만 대상으로 한다. 모델 전체 내용의 해시는 아직 수집하지 않았다. 개인키, 가상환경, 수집 데이터, 모델 대용량 파일은 인계 문서에 내장하지 않았다. Git 저장소는 아직 초기화하지 않았으므로 커밋 이력이 있다고 가정하지 않는다.
