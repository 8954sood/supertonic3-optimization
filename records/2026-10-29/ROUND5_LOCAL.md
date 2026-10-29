> 최신 사용자 지시: F1200을 기준 모델로 선택했고, 성공한 최적화의 커밋 및 기존 모델·필수 재현 자료 업로드를 다시 허용했다. 모델 축소/재학습은 중지하고 엔진만 개선한다. 아래 업로드 중지 문구는 당시 이력이다. 실제 서버 결과는 [6차 기록](ROUND6_ENGINE.md)에 이어진다.

> 최신 상태: [round5_state.json](round5_state.json). **교사는 원본 FP32 CFG4 50스텝만 사용한다. B 교사 재증류는 사용자 지시로 폐기했다.** 아래 B 교사 부분은 실패·폐기 이력이다. 모델 업로드·Git 커밋/푸시는 새 요청 전까지 금지한다.

# 5차: 로컬 품질 개선 및 사용자 청취

## 사용자의 최신 작업 지침

모델 업로드와 Git 업로드는 다시 명시적으로 요청할 때만 한다. 추가 Git 커밋·푸시·PR 생성과 Hugging Face 갱신을 중지한다. 개선 실험과 로컬 기록은 계속하며 사용자에게 실제 음질 비교를 요청한다.

지침 도착 직후 추가 HF 업로드 프로세스를 중단했다. 새 모델 commit 생성 전 LFS preupload 단계에서 KeyboardInterrupt로 종료했다. 공개 모델 revision을 읽기 전용으로 확인한 결과 여전히 `5d2903d742ac779f6b98f7f47f217c761f54e076`이고 `models/cfg1024-v2/`는 공개 파일 목록에 없다. 이미 전송된 미참조 LFS blob이 있을 수 있으나 새 HF 모델/데이터 릴리스는 완료하지 않았다. 재시도하지 않는다.

소스 `99557ea` 커밋은 이 지침 도착 전에 로컬에서 생성했으나 푸시하지 않았다. GitHub main의 기존 공개 재현 수정 `387f8c8`와 ARM PR 동기화는 앞서 완료된 작업이다. 이번4차 최적화 branch, 새 경량 모델, 새381MB 데이터 archive는 로컬에만 있다. 4차 문서의 추가 공개 계획을 실제 완료 상태로 읽지 않는다.

## 청취 비교 요청

`results/round5_listening_request.json`에 입력·음색·원본 WAV·결합 방법·해시를 기록했다. 모두 개발 validation 문장이다. A는 원본, B는 기존 선택, C는 추가2,000회 학습한1024-v2다. 같은 두 문장을 M1/F1 순서로 결합했고0.7초 무음을 넣었다. 음량 정규화·잡음 제거는 하지 않았다. 사용자에게 C의 음색/억양/자음/잡음 차이를 요청했다. 사용자는 C에 대해 “살짝 뭉개지고 기계음이 살짝 더 들어간 느낌”이라고 평가했다. 이어 동일 C 가중치의 반복부를 FP32로 바꾼 D도 “비슷하게 나쁨”이라고 평가했다. 정수 변환 해제만으로 해결되지 않으므로 과도한 축소/증류 손상을 우선 살핀다. 양자화 영향이 전혀 없다는 결론은 아니다.

## 다음 품질 실험

학생이 실제8회 생성할 때 쌓이는 오차를 줄이기 위해 `distill/train_rollout.py`로100회 짧은 후속 실험을 시작했다. 초기 가중치는 고정1024-v2의2000회 체크포인트다. 전체8스텝을 연결해 원본 최종 latent 및 고정 FP32 보코더의3해상도 STFT에 맞춘다. train만의 무작위 교사 update 손실0.1을 더해 원래 update 예측을 보존하는 정규화를 적용했다. LR2e-6, latent가중치1, STFT가중치0.15, gradient clip1,25회 간격 validation이다. 더 짧은 단계나 추론시 별도 후처리는 없다.

`--validation-all`이 과거8개로 고정되어 있던 부분을 manifest의 실제 validation case로 바꿔16개 전부 평가한다. 과거 실행의8개 검증 결과는 수정하지 않는다. 출력은 `distill/rollout-cfg1024-round5`, 로그는 `results/round5_rollout_train.log`다. 학습은MPS, 완성 모델의 추론은CPU1 조건이다. 이미 소비한 test는 다시 평가하지 않고 개발 검증과 청취로 판단한다. 손실 감소만으로 음질 개선을 주장하지 않는다.

## 추가 실험과 진행 상태

100회 latent 중심 rollout은 validation latent 상대MSE0.27001→0.25882였으나 MRSTFT1.29821→1.30151로 악화했다. 작은 수치 손실 개선을 음질 개선으로 채택하지 않았다. 결과는 `results/round5_rollout_latent_first.json`이다.

음성 손실 비중을 높인300회 별도 실험은 `distill/train_rollout_quality.py`다. latent0.25, MRSTFT1.0, active-frame2~10kHz log magnitude0.25, onset log-envelope difference0.1, teacher update0.1, LR3e-6을 썼다. validation audio loss로 선택한150회는1.571687→1.559188(약0.80% 감소), MRSTFT1.298212→1.286375였다. 최종300회 대신150회 best를 내보냈다. 이 정도로 사용자 피드백이 해결되었다고 주장하지 않는다. 실제CPU1 ONNX의 개발 ASR/음향 검사를 진행 중이며 test는 쓰지 않는다.

원인 진단: 같은 C 가중치에서 vector FP32/INT8만 바꾼4문장의 평균 정렬 logmel 차이는1.287dB, F0중앙 오차 평균10cents였다. 사용자는 FP32 D도 비슷하게 나쁘다고 판단했다. C/D를 기본 후보로 채택하지 않는다.

B를 직접 보존하는 부분 축소 방향으로 전환했다. 승인된 B INT8가 실제 생성한 궤적에서22개 넓은 FFN의 입력/출력을 train80·validation16문장·10음색으로 수집했다. 스텝당16행, train10,240행·validation2,048행/층이다. test는 수집하지 않았다. 학습 activation에서 neuron 중요도를 구하고, 남은 neuron의 출력 가중치를 원래 가중치에 ridge로 고정하며 재추정했다.

25% 폭 축소+gated validation 상대RMSE6% 기준은22개 중0개 통과해 제외했다. 더 가벼운 폭 축소(최대12.5%,1536층은128배수로1280)는2개 층만 통과했고, 학습 가중치45,480,256→44,955,456으로1.15% 감소했다. 속도 이득은 검증하지 않았다. 이 작은 감소를 실질적 속도 개선이라고 주장하지 않는다.

그다음 별도 medium 실험은25% 폭 축소 중 dev 상대RMSE12.5% 이내의12개 층만 사용한다. B의 전체 구조를 유지하고 나머지 층은 원래 폭/가중치 그대로 둔다. 이 예산 완화는 앞선 엄격 실험과 분리하며, 사용자 청취 전 기본값으로 바꾸지 않는다. 승인된 B FP32 vector를 고정 교사로 삼아 기존 원본CFG4 궤적 입력2,048개를 다시 라벨링하고 이 함수에 맞게 공동 학습한다. **B 자체 rollout을 새로 수집한 데이터가 아니라, 원본CFG4 state에 B 함수를 평가한 목표**다. 새 원본교사CFG4 증류라고 표기하지 않는다.

명령/경로: `distill/collect_balanced.py`, `distill/prune_balanced.py`, `distill/retarget_vectors.py`; 수집 `distill/balanced-activation-round5`, strict fit `distill/balanced-pruned-round5`, light fit `distill/balanced-pruned-light-round5`, medium 초기 `distill/balanced-pruned-medium-round5`, B 교사 `models/round5-balanced-fp32-teacher`, 재라벨 데이터 `distill/trajectories-balanced-round5`다. 이 단계들은 전부 로컬이며 추가 공개 업로드와 Git 커밋/푸시는 하지 않았다.

## 최우선 교사 정책 수정 — 사용자 지시

사용자가 “교사 모델은 무조건 원본 최대음질 모델을 써야 한다”고 지정했다. B를 교사로 삼은 재증류 방향은 잘못 잡았으며 폐기한다. B 대상800회 학습은 지시 도착 때 이미 완료되어 있었지만 평가 프로세스를 즉시 종료했다. 결과 모델을 교사·배포 후보·청취 후보로 사용하지 않는다. `models/round5-balanced-recovered`와 `models/round5-balanced-fp32-teacher` manifest에 REJECTED 상태를 기록했다. `distill/trajectories-balanced-round5`, B activation에 맞춘 strict/light/medium fit은 폐기한 실험 이력이며 새 학습에 재사용하지 않는다.

`distill/teacher_policy.py`를 추가했다. 학습기는 원본FP32CFG4 데이터만 받아들이며, student-retargeted 데이터와 파생 모델 교사를 거부한다. 기존 원본8스텝 데이터를 허용하되 새 데이터는 pinned upstream repo/revision, FP32, CFG4, non-distilled 및 원본 그래프 해시를 기록한다. 원본 수용·B 재라벨 데이터 거부를 실제 확인했다.

새 방향은 **원본 FP32 CFG4 50스텝 교사 → 실제8스텝 학생**이다. 50스텝이 언제나 지각상 더 좋다는 검증을 했다는 뜻은 아니다. 학습용 교사 생성만 Mac CPU4스레드로 수행한다. 이는 배포 학생의 CPU1 측정과 분리한다. 원본 ONNX 가중치를 변경하거나 INT8/학생 교사로 바꾸지 않는다.

`distill/collect_original_multistep.py`가 원본50회 궤적을 수집하고 k/8 시점에 선형 보간한 latent를 학생 입력/목표로 저장한다. 마지막 목표는 원본50회 최종latent 그대로다. 보간한 중간state는 원본교사8회 실행과 동일하지 않으며 명시적으로 구분한다. 고유train80문장·1음색seed/문장·640쌍, validation16문장·128쌍이며 test는 수집하지 않는다. 수집 경로는 `distill/trajectories-original50-round5`, 로그는 `results/round5_original50_collect.log`다.

학생 초기화는 `distill/init_partial_original.py`로 다시 만들었다. B 체크포인트는 학생 초기값으로만 사용한다. 교사 출력으로 쓰지 않는다. B의2048폭6개 층을 기존 **원본FP32 교사로만 학습한**1536 FFN으로 교체했다. 교체 층은 original-teacher validation fit이 가장 나았던21,26,8,14,18,17번이며, 나머지 가중치는 학생 초기값에서 유지한다. B target에 맞춘 ridge fit은 전혀 쓰지 않는다. 초기 경로 `distill/partial-original-round5`, 이후 모든 학습 목표는 원본50스텝이다.

### 원본50 청취 확인과 재현 명령

사용자는 원본50 M1/F1 비교 음성에 “원본은 괜찮음”이라고 답했다. 이 개발 샘플을 현재 청취 품질 기준으로 삼는다. 모든 음색/언어에 대한 MOS 검증이라는 뜻은 아니다. `results/round5_original50_listening.json`에 입력·음색·시드·해시·피드백을 보관했다.

```bash
python distill/init_partial_original.py --output distill/partial-original-round5
python distill/collect_original_multistep.py --output distill/trajectories-original50-round5
python experiments/audit_original_trajectories.py --data distill/trajectories-original50-round5 --output results/round5_original50_data_audit.json
python distill/train_joint.py --checkpoints distill/partial-original-round5/initial.pt --data distill/trajectories-original50-round5 --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx --train-all --iterations 1200 --learning-rate 1e-5 --validate-every 200 --output distill/joint-original50-round5
python distill/export_joint.py --checkpoint distill/joint-original50-round5/best.pt --graph models/cfg-student-fp32-source/onnx/vector_estimator.onnx --base models/fast-fp32-cached --int8 --cfg-distilled --output models/round5-original50-student
python distill/evaluate.py --corpus distill/corpus_round5_pilot.json --reference-steps 50 --variants models/cfg-mixed-v1-selected models/round5-original50-student --split validation --output results/round5_original50_pilot.json
```

기존 경로가 있으면 이 명령을 다시 실행하지 말고 진행 로그·완료 여부부터 확인한다. 재실행 시 새 출력 이름을 선택한다. 전부 로컬 작업이며 업로드를 포함하지 않는다. 자동 순차 실행 로그는 `results/round5_original50_pipeline.log`다. 새 pilot은 개발 검사이며 독립 시험이 아니다. 원본 참조는50스텝, 학생은8스텝으로 명시한다. 과거8스텝 참조의 음향 거리와 그대로 비교하지 않는다.

### 원본50 데이터 확인 및 중간 청취 후보

원본50 수집이 완료됐다. train80문장/640쌍, validation16문장/128쌍이다. 전체768개 NPZ의 유한값·8스텝 메타데이터·672개 인접 state 연결이 모두 통과했다. `results/round5_original50_data_audit.json`에 source 그래프 해시와 데이터 manifest 해시를 기록했다.

원본-only partial 학생을1200회·LR1e-5·200회 검증으로 학습한다. 초기 validation update MSE는0.0697527이다. 원본50을 교사로 한 데이터만 로드했고 source-policy 검사를 통과했다. 400회 시점 best 체크포인트를 별도 복사해 `models/round5-original50-preview`로 내보냈다. 이것이 사용자에게 새로 들려준 **E**다. C/D와 다르며, 폐기된 B-교사 모델과도 다르다. 실제학생8회, 단일conditional경로, 반복부 reduced-rangeINT8, frontend/vocoder FP32다. 별도 후처리·음량 정규화를 하지 않았다.

E의 M1/F1 두 개발 문장 대 원본50의 logmel 차이는3.624/3.866dB, F0문장별중앙차이는20/20cents였다. 지각상 개선 확정이나 음질 일치율이 아니다. `results/round5_original50_preview_listening.json`의 체크포인트 해시와400회 step이 그 청취 대상이다. 최종1200회 모델에 피드백을 옮기지 않는다.

### 이번 반복 완료 상태

원본50→8 학생1200회 학습과4문장(KO2/EN2) 개발 pilot을 완료했다. validation update MSE0.0697527→0.0421980, optimizer loop 약216.3초다. MSE 감소율을 음질 개선율로 해석하지 않는다.

| 같은 원본50 참조·개발4문장 | KO CER | EN WER | 평균 정렬 logmel 거리 |
| --- | ---: | ---: | ---: |
| 원본50 | 0/44 | 0/20 | 0 |
| 기존 B | 0/44 | 0/20 | 3.923dB |
| E400 | 0/44 | 0/20 | 3.834dB |
| F1200 | 0/44 | 0/20 | 3.665dB |

작은 개발 pilot이며 독립시험·MOS·원본 일치율이 아니다. 실제8스텝 학생만 비교했다. 참조 원본50은50스텝이다. 반복부 ONNX 파일은 B52,077,304바이트→E/F48,903,928바이트로 줄었다. 이것을 같은 비율의 추론 속도 향상으로 주장하지 않는다. 새 후보의1GB 서버/ARM 속도는 아직 측정하지 않았다.

사용자는 **E400**의 M1/F1 두 문장에 “기존 B 수준에 가까움”이라고 답했다. E400을 청취 확인된 개발 기준으로 고정한다. 기본 B를 자동 교체하지 않는다. F1200은 개발 지표가 더 가까워졌지만 아직 사용자가 듣지 않았다. `samples/listening_original50_preview/F-original50-final_M1_F1.wav`를 같은 문장/음색/seed로 만들었고, 그 피드백을 다음에 받는다. E 피드백을 F에 옮기지 않는다.

현재 집계는 `results/round5_summary.json`, 원본50 학습 보고서는 `results/round5_original50_training.json`, 새 청취 대상/해시는 각 listening JSON에 있다. 모델·Git 추가 업로드와 커밋/푸시는 하지 않았다. 다음 작업은 F와 E의 실제 청취 비교 후 후보 고정, 더 넓은 개발 검사, 조건이 허용될 때 새 모델의 실제서버 검증이다. 이미 소비한 test는 튜닝에 재사용하지 않는다.
