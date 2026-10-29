# 4차: 공개 업로드 검증, 추가 추론 실험, 경량 모델 재학습

기본 선택은 **cfg-mixed-v1-selected**를 유지한다. 새 `cfg1024-v2`는 실제 8스텝·CPU1에서 더 빠르지만 독립 시험의 한국어 CER가 이전 1024 모델보다 나빠졌다. 음향 거리의 개선과 작은 검증 세트만으로 교체할 근거가 부족하다. 새 모델과 필수 초기/최종 체크포인트·교사 궤적은 재현용 실험 자료로 분리한다. 기존 청취 의견을 새 모델의 승인으로 옮기지 않는다.

## 완료한 공개·재현 작업

[모델](https://huggingface.co/babihoba/supertonic3-optimization) 최초 완전 검증 revision `85bc100ca8d7cefad17b63a4c7ba489638dbe2f2`에서 67개 원본 자산, [데이터](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data) revision `817056ae3e9b8fa1a2e464a212bb2ba1f4c5196a`에서 1,665개 자산의 원격 크기와 내용 해시를 전부 확인했다. 이후 카드·LICENSE만 수정했다. 공개 다운로드에는 인증을 쓰지 않는다.

- 선택 ONNX 공개 다운로드 후 기존 로컬 모델과 파형이 비트 단위로 같았다.
- 공개 초기 체크포인트와 공개 NPZ를 옮긴 경로에서 읽고 실제 AdamW 1회 업데이트를 수행했다.
- 공개 600회 체크포인트에서 다시 내보낸 네 ONNX 파일 해시가 기존 배포본과 전부 같았다.
- 공개 궤적에 대한 PyTorch/ONNX 벡터 8개·decoder 4개 parity를 확인했다.
- `train_joint.py`가 manifest의 과거 저장 위치 대신 현재 `--data` 아래 split/파일명을 해석하도록 수정했다. 다운로드·검증 도구와 재현 명령은 main `387f8c8`에 반영했다.

근거는 [공개 완료 기록](PUBLICATION.md), `results/round4_hf_upload_verification.json`, `round4_public_download_verification.json`, `round4_public_reproduction_verification.json`, `round4_reexport_verification.json`, `round4_published_graph_parity.json`이다.

## 사용자 제공 ARM 측정 PR

[PR #1: Record Oracle ARM RTF measurements and compare 1/2-thread inference](https://github.com/8954sood/supertonic3-optimization/pull/1)을 검토했다. Neoverse-N1 2코어 약12.5GB, ORT1.31.1, 선택 모델 revision 85bc100의 KO/EN 짧은·긴 4문장이다. 한 스레드 RTF 0.276~0.283, 두 스레드 0.153~0.161이다. 같은 네 문장에서 1/2스레드 PCM16 WAV는 같았다. 음질·ASR 평가가 추가된 것은 아니다.

공개 다운로드 안내와 생긴 README 충돌은 main을 PR 브랜치에 합쳐 해결했다. PR은 draft를 유지하며 병합하지 않았다. 이 작업에서는 ARM 서버에 접속하지 않았고, 새 모델의 ARM 속도를 추정하지 않는다. 사용자 요청대로 추가 서버 평가는 기존 x86 1코어·1GB에서만 수행했다.

## 추론 그래프 실험

`experiments/round4_variants.py`는 기존 가중치를 보존하고 새 폴더를 만든다. 공통 파일은 hardlink로 공유하되 모든 변경은 새 파일 교체 방식이다. source manifest와 실제 새 그래프 해시를 별도로 기록한다. 새 모델에 기존 청취 승인을 붙이지 않는다.

| 실험 | 실제 수행 | 결과·결정 |
| --- | --- | --- |
| 학생 조건 캐시 | 반복부 1,510→1,316노드, 불변 194노드·19텐서 재사용 | 로컬 4음색/문장 파형 완전 일치. 서버 KO5.772/EN5.636초로 뚜렷한 양방향 이득 없음 |
| 보코더 마지막 head INT8 | 마지막 2048×512 MatMul 하나, per-channel reduced-range | 검증 ASR 유지, 기존 선택 대비 logmel1.457dB·SNR38.44dB. 서버5.820/5.418초, 채택하지 않음 |
| 벡터·보코더 FastGELU | 28+10개 활성화 근사 | 검증 ASR 유지하지만 파형 SNR7.86dB. 서버5.905/5.309초, 채택하지 않음 |
| 보코더만 FastGELU | 10개 활성화 근사 | 선택 대비 logmel0.0406dB·SNR52.68dB. 교차 재측정에서 KO 유의미한 이득 없음 |
| head INT8 + 보코더 FastGELU | 위 두 보코더 변환 결합 | 서버5.801/5.374초. 유의미한 추가 이득 없음 |
| head 저랭크 가능성 조사 | SVD 에너지, 실제 모델 내보내기 없음 | rank384에서99.18% 에너지지만 곱셈 감소6.25%뿐. 재학습 없는 축소는 진행하지 않음 |

그래프 후보의 서버 값은 warm-up+3회 중앙값, 공유 서버·420MB RSS 감시/80MB available reserve·CPU affinity0이다. 같은 sweep의 기준 KO7.973초는 다른 paired 실행의5.823~5.911초보다 느렸다. 따라서 **7.973초를 분모로 큰 개선을 주장하지 않는다**. 보코더 FastGELU를 별도로 3쌍 교차 측정했을 때 KO paired speedup 중앙값1.001배였다. EN은1.093배지만 8.399초 기준 outlier가 있고 KO 이득은 없으므로 기본 모델 변경 근거로 삼지 않았다. 모델별 RSS 차이도 allocator/프로세스 변동을 고려한다.

로컬 M4 Max profiling은 분석용이다. 벡터 시간은 DynamicQuantizeMatMul29.3%, LayerNorm28.6%, GELU18.2%였다. 보코더는 MatMul27.3%, FusedMatMul20.3%, BiasGelu21.2%, LayerNorm16.9%였다. 이를 Oracle ARM의 연산 비율로 환산하지 않는다. 독립 PDN 엔진이나 Paradee 방식 DSP head를 구현한 결과가 아니다.

## 기존 모델 추가 시험

추가로 고정한 첫 번째 16문장 `distill/corpus_round4.json` test에서 원본 KO9/209·EN0/105, 선택 모델8/209·0/105, 이전1024 모델9/209·1/105, CFG4 혼합 폭10/209·2/105였다. 이 세트는 그 시점에 소비했고 새 학습의 독립 시험으로 재사용하지 않았다. 원시 결과는 `round4_baseline_holdout.json`이다. 과거 test-v3의 선택 모델 EN3/69 결과도 그대로 유지한다.

## 1024 폭 추가 증류

교사는 고정 원본 FP32 CFG4, 실제8스텝이다. 새 텍스트는 직접 작성했으며 원본 train40개에 KO/EN20개씩을 추가해 고유 train80개를 만들었다. 음색/seed3조합을 적용한240case·1,920교사 쌍, validation16문장·128쌍이다. 학습 중 update MSE 검증에는 step0/3/7의48쌍을 썼다. 이48쌍을 전체 validation128쌍 또는 독립 test와 혼동하지 않는다. 수집은 CPU1, 학습은 Mac MPS이다.

초기 가중치는 이전 폭1024 모델의 2,400회 체크포인트다. 전체 벡터 가중치를 AdamW lr1e-5로 추가2,000회,200회 간격 검증했다. 추가 optimizer loop 약294.5초, validation normalized update MSE **0.0757166→0.0699235**이다. 보코더와 frontend는 FP32 원본 그대로다. checkpoint400과2000을 dev 비교한 뒤2000을 고정했다.

| 같은 dev16문장 | KO CER | EN WER | 원본 대비 평균 DTW logmel |
| --- | ---: | ---: | ---: |
| 원본 | 2/193 | 0/96 | 0 |
| 기존 선택 | 2/193 | 0/96 | 3.875dB |
| 이전1024 | 6/193 | 0/96 | 4.436dB |
| 추가400회 | 3/193 | 0/96 | 4.479dB |
| 추가2000회 | 3/193 | 0/96 | 4.312dB |

`cfg1024-v2`는 이2000회 후보를 시험 전에 고정한 것이다. 벡터 SHA-256은 `9fd169498fcedd9c52f8c6b3092c56703c6115687b868f1ee8c89d1bdb2be133`이다. 더 짧은 스텝, 가지 생략, 동일 출력 캐시는 사용하지 않는다.

## 새 독립 시험: 개선이 일반화되지는 않음

`distill/corpus_round4_train.json`의 test는 학습 전에 새로 고정한 KO8/EN8문장이다. train/validation 및 앞선 모든 test와 정규화한 텍스트 중복이0이다. 체크포인트 선택을 마친 뒤 네 모델을 한 번 평가했다. 이제 이 세트도 소비했으므로 후속 학습 선택에는 쓰지 않는다.

| 같은 독립16문장 | KO CER | EN WER | 원본 대비 평균 DTW logmel | 문장별 F0 오차 중앙값의 평균 |
| --- | ---: | ---: | ---: | ---: |
| 원본 FP32 | 7/218 (3.21%) | 2/109 (1.83%) | 0 | 0 |
| 기존 선택 | 9/218 (4.13%) | 2/109 (1.83%) | 4.093dB | 45.94cents |
| 이전1024 | 6/218 (2.75%) | 2/109 (1.83%) | 4.813dB | 64.69cents |
| **추가학습1024-v2** | **10/218 (4.59%)** | **2/109 (1.83%)** | **4.460dB** | **46.25cents** |

영어의2오류는 모든 모델에서 같은 `streetlights`/`street lights` 분리 때문에 발생했다. 유리하게 재채점하지 않고 기존 정규화 결과를 남겼다. 한국어는 새 경량 후보가 dev에서는 나아졌으나 독립 시험에서는 나빠졌다. logmel·F0 거리는 이전1024보다 개선되었지만 기존 선택보다 좋은 모델이라고 판단하지 않는다.

Whisper base INT8·CPU1·beam5는 발음 대리 검사다. 한국어 공백 제거 CER, 영어 단어 WER이며 MOS·화자 유사도·청취 일치율이 아니다. F0는 pYIN 추적 오차가 포함될 수 있고 DTW 정렬은 원형 파형 일치가 아니다. Paradee의 harmonic 수치와 커스텀 지표도 직접 비교하지 않는다.

## 1코어·1GB 실제 서버 교차 측정

같은 EPYC7551 호스트에서 기본 선택 모델과 새1024-v2를 순서를 바꾸며3쌍 측정했다. 각 프로세스는 자기 warm-up 후1회 측정하고 종료한다. 동시에 두 모델을 메모리에 두지 않는다. CPU affinity0, ORT/BLAS1스레드, nice19, 실제8스텝,50ms 표본 RSS420MB·available reserve80MB다. 로딩·파일 쓰기는 제외한다.

| 짧은 문장 | 기본 선택 중앙값 | 1024-v2 중앙값 | 쌍별 가속비 중앙값 |
| --- | ---: | ---: | ---: |
| KO, 음성5.320초 | 5.911초 | 4.809초 | 1.247배 |
| EN, 음성4.934초 | 5.611초 | 4.373초 | 1.292배 |

전체6회 중 표본 peak RSS는 선택416.69MB, 새 후보395.58MB였다. 최소 available은238.76MB 이상이었다. 모든 감시 실행이 성공했다. 쌍별 가속비와 개별 중앙값의 비율은 서로 다른 통계다. 공유 서버의 첫 기준 실행이 느렸으므로3쌍의 가속비 범위도 원시 JSON에 보존한다. 이전1024 대비 추가 속도 개선을 검증한 것은 아니다. 새 후보는 **기존 경량 모델과 같은 크기에서 품질 회복을 시도한 실험**이다.

서버가 실제 만든 KO/EN 두 WAV를 로컬로 가져와 ASR했다. 기준과 새 후보 모두 KO0/25·EN0/11이었다. 두 짧은 문장 smoke check이며 독립16문장의 회귀를 무효화하지 않는다. 새 후보의 ARM 측정과 청취는 아직 하지 않았다.

근거: `results/round4_paired_expanded.json`, `round4_epyc_candidate_asr.json`, `round4_epyc_reference_asr.json`. 전체 원시 파일 색인과 집계는 [round4_metrics.json](round4_metrics.json), [공개 결과 해시](../public_source_index.json)에 있다.

## 공개할 최소 추가 자료

실험용 `models/cfg1024-v2`, 초기1024 체크포인트1개, 선택2000회 체크포인트1개, 설정·학습 보고서, 직접 작성한 corpus, train/validation 궤적만 공개한다. 중간400회·실패 그래프 모델은 올리지 않는다. 모델/초기/최종 논리 합계 약401MB, 새 궤적 압축 약381MB다. 기존 자산과 같은 가중치는 Hub가 중복 저장을 줄일 수 있다.

새 데이터는2,049개 파일을 하나의 tar.gz로 묶었다. archive의 uid/gid는0, 소유자명은 빈 값, mtime0, 권한0644이며 상대 경로의 manifest/NPZ만 포함한다. 압축 파일과 각 추출 파일의 SHA-256을 함께 보관한다. 새 업로드의 고정 revision과 다운로드·재현 명령은 [재현 안내](../../docs/REPRODUCING.md)에 추가한다. 기본 모델의 기존 고정 revision은 유지한다.
