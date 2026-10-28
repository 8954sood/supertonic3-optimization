> 당시 시점의 기록입니다. 최신 확정 상태는 [HANDOFF_ROUND3.md](HANDOFF_ROUND3.md)를 참조하세요.

# Supertonic 3 최적화 작업 기록

기록 시점: **2026-10-28 16:55 KST**. 사용자의 “여태까지 한 것부터 기록” 요청에 따라 기존 파일을 취합했다. 이 기록을 만들면서 새 추론·학습은 시작하지 않았다.

## 현재 판단

**속도 개선은 확인했지만 음질 목표는 달성하지 못했다.** 사용자가 원본과 다르게 어색하다고 직접 청취 의견을 주었다. 어느 WAV를 들었는지 파일명이 특정되지 않았으므로 특정 변환 하나를 원인으로 단정하지 않는다. 현재 손실 변환·증류 후보를 완성품으로 취급하지 않는다. ASR가 맞아도 음색·억양·자연스러움이 같다는 뜻이 아니다.

가장 빠르게 실제 서버에서 측정한 원래 CFG4 유지 후보는 `models/student1024-int8`: 한국어 **7.4707초**, 영어 **7.3980초**, 최대 표본 RSS 약 **259MB**다. 약 5.3~5.5초 음성을 만드는 데 걸린 시간이므로 **아직 실시간보다 느리다**. 폭 512 모델은 발음 지표가 악화돼 현재 설정으로 채택하지 않는다. 폭 256은 층별 학습만 끝났으며 전체 음성·속도를 검증하지 않았다.

원본 음질 보존 기준은 `models/supertonic-3`; 수치적으로 원본에 가까운 실행 최적화 후보는 `models/fast-fp32-cached`다. 후자는 작은 서버에서 메모리 감시기에 중단됐으므로 해당 서버의 유효 속도 결과가 없다.

## 요구 조건과 측정 환경

- 실제 denoising **최소 8회**. `runtime.py`가 8 미만을 거부한다. 후반 가이던스 축소 실험도 denoising 호출은 8회지만 알고리즘은 변경한다.
- 추론 CPU 1스레드: ORT intra/inter=1, sequential, CPUExecutionProvider, BLAS/OpenMP도 1스레드. 서버 CPU/wall은 약 1.0.
- 원래 목표: Oracle ARM 2코어·12GB. **ARM 실측 없음**.
- 실제 제공 서버: Oracle **AMD EPYC 7551 x86_64**, 물리 1코어·논리 2CPU, RAM 약 954MB, Linux/Python 3.12.3. 논리 CPU 0에 고정, nice 19. 다른 프로그램과 공유한다.
- Mac: Apple M4 Max. Mac 수치를 Oracle 성능으로 환산하지 않는다. 학생 모델 훈련에는 MPS GPU를 사용했고 추론 성능에는 포함하지 않았다.
- 서버에서는 격리 폴더·가상환경만 사용. 시스템 서비스와 다른 프로그램을 변경/종료하지 않았다. 메모리 감시기는 50ms 표본 감시이며 OS 강제 상한이 아니다.
- 원본 모델: `Supertone/supertonic-3`, 리비전 `3cadd1ee6394adea1bd021217a0e650ede09a323`. 원본 OpenRAIL-M 라이선스를 패키지에 유지한다.

## 시도한 방법과 결론

| 순서 | 시도 | 구현/기록 | 관찰과 판단 |
| --- | --- | --- | --- |
| 1 | 원본 1스레드 실행기와 공식 출력 대조 | `runtime.py`, `tests_runtime.py` | KO/EN 8스텝에서 공식 core 출력과 일치. 초기 1~5스텝 프로파일은 이후 8스텝 속도 비교에서 제외 |
| 2 | 기존 MatMul만 INT8 | `optimize.py`, `results/matmul_only_int8_8.json` | 주요 병목 Conv가 남아 개선 제한 |
| 3 | 문장 조건 캐시 | `optimize.py`, `prepare.py` | 매 스텝 불변 176노드·19경계 텐서 분리. 단독 개선은 작았음 |
| 4 | 1×1 Conv를 MatMul로 변환 | `experiments/pointwise.py` | 총 140개 변환. FP32+조건 캐시는 Mac 교차 측정에서 4.54~4.71배 |
| 5 | 반복부/전체 동적 INT8 | `prepare.py`, `models/fast-*int8` | 작은 서버에서 전체 INT8 실행 가능. 정밀도 손실과 파형 차이 존재 |
| 6 | 정적 QDQ 보정 | `experiments/static_quantize.py` | 실제 8스텝 궤적을 보정에 사용했지만 서버에서 더 느림 |
| 7 | AVX2용 reduce_range | `experiments/pointwise.py`, `models/fast-avx2-int8` | U8S8 포화 완화용 [-64,+64] 가중치 범위. 서버 성능의 큰 개선은 없음 |
| 8 | ORT 형식·메모리 옵션·구버전 | `experiments/ort_format.py`, `results/guard_*` | FP32는 가용 RAM 부족으로 중단. ORT 1.20.1은 Q8 속도 큰 변화 없음 |
| 9 | INT4 MatMulNBits | `experiments/int4.py` | 저장/메모리는 감소했지만 서버 속도 악화 |
| 10 | 학습 없는 MLP 채널 제거 | `experiments/prune_expansion.py` | 발음 손상. 반폭·3/4폭 후보 거절. 학습 없는 제거는 기본 후보에서 제외 |
| 11 | PDN 공개 번들 분석 | `references/pdn-notes/inspection.json` | SIMD·packed weights·arena·낮은 프레임률 DSP를 참고. 받은 ZIP에는 JS/WASM만 있었고 C 원본은 없었음 |
| 12 | Python/NumPy BLAS 전용 연산 | `numpy_ops.py`, `experiments/numpy_backend.py` | 서버 약 50초로 악화 |
| 13 | 자체 C + OpenBLAS 연산 | `native/`, `native_ops.py` | 연산 수치 검사 통과. 약 262MB지만 약 15초로 표준 Q8보다 느림. 독립 엔진이 아닌 ORT 내부 전용 연산 |
| 14 | CFG의 동일한 앞부분 중복 제거 | `experiments/guidance.py dedup` | CFG4 유지, 검사한 짧은 두 음성은 Q8 기준 비트 일치. 서버 속도는 일관된 개선 없음 |
| 15 | 후반 CFG4→CFG1 | `experiments/guidance_schedule.py` | 4+4스케줄+중복 제거 약 9초. 원본 알고리즘·음색/억양 변경 가능. 원본 품질 유지 후보가 아님 |
| 16 | 교사 활성값 수집과 실제 층별 증류 | `distill/collect.py`, `train.py`, `export.py` | 폭 1024/1536/512/256 훈련 완료. 1024는 서버 약 7.4초. 512는 발음 악화. 음질 개선 작업 미완료 |

## 실제 서버 성능

표의 값은 워밍업 후 전체 파형 생성 시간의 중앙값이다. 서로 다른 시점의 공유 서버 측정이므로 작은 차이를 유의미한 개선으로 판단하지 않는다. 프로파일링/단발 측정도 구분한다.

| 후보 | KO 초 | EN 초 | 반복 | 대략 RSS | 원시 파일 |
| --- | ---: | ---: | ---: | ---: | --- |
| 전체 INT8 | 12.1167 | 11.8015 | 2 | 322MB | `results/epyc_full_int8_8.json` |
| 정적/동적 혼합 INT8 | 15.0646 | 미측정 | 1 | 감시 기록 참조 | `results/epyc_hybrid_static_8.json` |
| INT4 | 16.0663 | 15.7976 | 2 | 282MB | `results/epyc_int4_8.json` |
| Python BLAS | 50.0867 | 미측정 | 1 | 341MB | `results/epyc_blas_8.json` |
| C/OpenBLAS | 15.0120 | 14.5906 | 2 | 262MB | `results/epyc_native_blas_8.json` |
| ORT 1.20.1 Q8 | 11.8056 | 미측정 | 1 | 감시 기록 참조 | `results/epyc_ort120_int8_8.json` |
| CFG 앞부분 중복 제거 | 12.6447 | 11.1487 | 2 | 312MB | `results/epyc_dedup_int8_8.json` |
| 후반 CFG 축소 + 중복 제거 | 9.0350 | 8.9645 | 5 | 311MB | `results/epyc_guidance_four_repeat5_8.json` |
| **폭 1024 증류 INT8** | **7.4707** | **7.3980** | **3** | **259MB** | `results/epyc_student1024_8.json` |

1024 증류는 첫 전체 INT8 측정보다 소요 시간이 KO 약 38.3%, EN 약 37.3% 줄었다. **12초 기준은 원본 FP32가 아니다.** 원본 FP32/FP32 ORT는 RAM 여유 부족으로 중단되어 속도 비교 불가다. 감시 결과는 `results/guard_*.json`에 있다. 일부 오래된 C 연산 측정의 `backend` 문자열은 ORT kernels로 남아 있으므로 모델 이름/manifest와 함께 읽어야 한다. 현재 벤치마크 코드는 이를 수정했다.

Mac의 수학적 연산 보존 FP32+캐시 교차 측정(각 3회)은 KO 짧은 문장 1.623→0.357초, KO 긴 문장 5.206→1.138초, EN 짧은 문장 1.564→0.342초, EN 긴 문장 5.090→1.080초다. `results/paired_fp32_cached_8.json`과 `results/summary.md` 참조.

## 발음 지표와 원본 대비 일치

**WER/CER가 같아도 소리가 같다는 뜻은 아니다.** KO는 공백 제거 문자 CER, EN은 단어 WER다. 아래는 분리한 검증 8문장(KO4/EN4), 6개 음색 순환, 문장별 고정 seed, 8스텝 결과다. 이 검증 세트는 후보 선택에 사용했으므로 독립 최종 시험이 아니다.

| 후보 | KO 오류/문자(CER) | EN 오류/단어(WER) | KO 원본 ASR와의 편집 일치 | EN 원본 ASR와의 편집 일치 |
| --- | ---: | ---: | ---: | ---: |
| 원본 FP32 | 2/88 (2.27%) | 0/46 (0%) | 기준 | 기준 |
| 1024 증류 후 INT8 | 2/88 (2.27%) | 0/46 (0%) | 98.86% | 100% |
| 1024 QAT 후 INT8 | 2/88 (2.27%) | 0/46 (0%) | 98.86% | 100% |
| 512 증류 후 INT8 | 12/88 (13.64%) | 3/46 (6.52%) | 86.36% | 93.48% |
| 512 QAT 후 INT8 | 10/88 (11.36%) | 4/46 (8.70%) | 87.50% | 91.30% |

원본 ASR 편집 일치 = `1 - edit_distance(원본 음성의 인식문, 후보 음성의 인식문) / 원본 인식문 길이`. 정확히 같은 문장 비율이나 음향 유사도, 화자 일치율이 아니다. 정답 문장에 대한 CER/WER와 구별해야 한다. 1024의 한국어 한 문장은 원본과 후보 모두 2문자 오류지만 서로 다른 인식문이므로 ASR 편집 일치는 100%가 아니다.

원시 자료: `results/distill_validation1024.json`, `results/distill_validation512.json`. 실제 서버에서 생성한 1024의 벤치마크 두 WAV도 ASR 오류 0이었다(`results/asr_epyc_student1024.json`). 1024 FP32 개발용 KO3/EN3도 오류 0이었으나 그 6문장은 여러 후보에 재사용했다. 이전 JSON의 “heldout” 표현은 독립 시험이라는 의미로 사용하면 안 된다.

원본 대비 FP32+캐시의 짧은 KO/EN 파형 상관은 각각 0.9999999996/0.9999999987, RMSE 1.28e-6/2.42e-6이다. 다른 음색·seed·8/12스텝 검사 4조합 RMSE는 6.8e-7~5.8e-6이었다. `results/equivalence_fp32_cached.json`, `tests_optimized.py` 참조.

INT8부터는 파형이 달라진다. 전체 INT8은 길이 예측도 바뀌어 시점별 파형 상관이 약 0에 가까워질 수 있다(`results/quality_int8.json`). 길이가 다른 파형을 앞에서 맞춘 단순 상관은 음질 점수로 사용할 수 없다. 반복부만 INT8인 짧은 두 경우의 상관은 약 0.688/0.545였다(`results/quality_denoiser_int8.json`). 학생 모델의 정렬된 스펙트럼·F0·음색 지표는 아직 수집하지 않았다. 수치가 없는데 “음질 일치율”을 만들어 제시하지 않는다.

## 증류의 실제 범위

- 훈련 40문장(KO20/EN20), 검증 8문장, 별도 최종 시험 12문장. 시험 12문장은 아직 사용하지 않았다.
- 10개 원본 음색으로 원본 FP32 CFG4·8스텝 궤적을 수집. 층마다 훈련 30,624행/검증 6,144행. float16 저장·float32 교사 연산.
- 28개 FFN만 2048→1024/1536/512/256으로 축소. 잔차512·attention·8회·CFG4 유지. ridge 출력층 재적합→AdamW 층별 증류→최종 QAT 비교.
- 층별 교사 입력으로 학습했다. 전체 학생 궤적을 통해 미분하는 end-to-end 학습, 파형/다중해상도 STFT/피치/위상/적대적 손실 훈련은 **아직 하지 않았다**.
- 폭1024: 1600회/층, 마지막600회 QAT. 반복부 파라미터 64,014,576→34,625,776(약46% 감소). 전체 패키지가 46% 감소한 것은 아니다.
- 폭1536: 1600/600, 모델 내보내기까지 완료, 음성·서버 미검증.
- 폭512: 2400/600, FP32가중치/QAT가중치 각 INT8 내보내기·검증 완료, 현 설정 발음 악화로 제외.
- 폭256: 3200/800, 28층 학습 완료, 아직 내보내기·전체 음성 미검증.
- 이는 Paradee의 12,000문장·23.9시간 교사 오디오 훈련 규모와 다르다. 지금의 40문장 활성값 증류를 그 수준의 음질 학습으로 설명하면 안 된다.

## 아직 측정하지 않은 것

청취 MOS/선호도, 화자 임베딩 유사도, DTW 정렬 스펙트럼 거리, 유성구간 F0 오차/유무성 일치, HNR 및 2~4kHz 배음/중간대역 비율, UTMOS, 장문·나머지 언어, Oracle ARM 실측. 현재 학습은 보코더를 증류하거나 Paradee식 새 DSP 헤드를 학습한 것이 아니다.

## 기록 파일

- `metrics.json`: 기존 67개 성능 행, 60개 ASR 집계, 파형 지표, 교사 대비 인식문 편집 일치, 5개 학습 실행(파일럿 포함)의 층별 오차.
- `source_index.json`: 취합한 원시 JSON 경로·SHA-256. 이후 파일이 바뀌었는지 확인 가능.
- `model_inventory.json`: 현재 32개 모델 폴더 크기·manifest. 실험 실패 모델도 보존.
- `HANDOFF.md`: 다음 작업자가 읽을 실행 방법, 상태, 주의점, 다음 실험 순서.
- 원시 결과는 `results/`, WAV는 `samples/`, 구현은 프로젝트 루트·`experiments/`·`distill/`·`native/`에 보존. 개인키는 기록 내용이나 패키지에 복사하지 않았다.
