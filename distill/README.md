> 아래는 초기 층별 증류 기록입니다. 현재 혼합 폭·CFG 증류 학습과 FP32 frontend/decoder 구성은 [최신 재현 안내](../docs/REPRODUCING.md)를 따릅니다.

# 8스텝을 유지하는 층별 증류

원본 Supertonic 3의 반복 추론부에서 28개 GELU MLP의 내부 폭 2048을 줄인다. 512채널 잔차, attention, CFG4, 실제 denoising 호출 8회는 유지한다. 출력 길이 예측기·텍스트 인코더·보코더는 기존 INT8 패키지를 사용한다. 작은 모델의 재학습 실험이며, 8M 규모의 새 음성 모델을 처음부터 학습한 결과는 아니다.

훈련은 Mac MPS GPU에서 수행한다. 추론과 서버 측정은 CPU 한 스레드다. GPU 훈련 시간을 CPU 추론 성능으로 제시하지 않는다.

## 데이터와 학습

`corpus.json`은 직접 작성한 훈련 40문장(KO/EN 각 20), 검증 8문장, 별도 시험 12문장으로 나뉜다. 10개 원본 음색을 교대로 사용하여 원본 FP32 교사의 실제 CFG4·8스텝 궤적에서 층별 입출력을 수집했다. 층마다 훈련 30,624개와 검증 6,144개 토큰 행을 저장했다. **30,624개 문장이나 독립 음성 녹음이 아니다.** 교사 출력의 합성 데이터만 쓴 작은 실험이다.

1. 원본 잔차 gamma와 가중치 크기로 초기 뉴런을 선택한다.
2. 8,192개 훈련 행으로 출력 선형층을 ridge 회귀한다.
3. 교사 FFN 출력을 목표로 모든 학생 FFN 가중치를 AdamW로 학습한다.
4. 최종 구간에는 동적 U8 활성값·축별 감소 범위 INT8 가중치를 모사한 QAT도 비교한다.
5. 검증 오차가 가장 작은 FP32/QAT 가중치를 별도로 저장한다.

QAT의 가짜 양자화는 미니배치 범위로 계산하므로 ORT의 전체 문장 동적 범위와 완전히 같지 않다. 최종 ONNX 음성을 반드시 따로 검사한다. 층별 교사 입력으로 훈련하므로 학생 전체의 누적 오차를 직접 최적화한 end-to-end 학습도 아니다.

## 재현

```bash
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python distill/collect.py
.venv/bin/python distill/train.py --output distill/student1024 --width 1024 --iterations 1600 --qat-steps 600
.venv/bin/python distill/export.py --checkpoints distill/student1024 --output models/student1024-int8 --state float --int8
.venv/bin/python distill/evaluate.py --variants models/student1024-int8 --split validation --output results/distill_validation.json
```

`collect.py --help`에서 데이터 경로·샘플 수를 확인한다. 메모리매핑 데이터는 약 2.2GB다. 학습 스크립트는 완료된 동일 설정 층은 건너뛰고, 미완료 층은 처음부터 재시작한다. 체크포인트는 100회 갱신마다 원자적으로 저장한다. 내보내기는 기존 폴더를 덮어쓰지 않는다. `--max-rmse`를 지정하면 검증 오차가 큰 층은 원본 폭으로 보존할 수 있다.

검증 문장으로 후보를 고른 뒤 별도 `test`를 한 번 사용한다. 기존 `experiments/asr_check.py`의 6문장은 여러 실험에서 재사용한 개발용 검사다. 음성 인식 결과는 음색·억양·자연스러움이나 학습하지 않은 언어의 품질을 보장하지 않는다. 배포 품질 판단에는 청취 평가와 더 많은 실제 입력이 필요하다.

원본 모델 라이선스는 내보낸 패키지의 `LICENSE`에 유지한다. 수집 데이터·체크포인트·개인키는 배포 패키지에 넣지 않는다.
