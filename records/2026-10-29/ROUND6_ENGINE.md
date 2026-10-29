# 6차: 모델 고정, 실제 1코어·1GB 서버 엔진 최적화

사용자가 성공한 최적화의 커밋·GitHub 푸시와 기존 모델/재현 자료 업로드를 다시 허용했다. F1200을 고정 기준으로 선택했으며 모델 구조·가중치는 변경하지 않는다.

## 이번 지침과 고정 조건

사용자는 E400·F1200에 큰 청취 차이가 없다고 답했고 모델 축소를 중지하도록 지정했다. 이후 작업은 같은 F1200 가중치·양자화·음색·시드·속도·실제 denoising 8회를 유지한 엔진 변경이다. E/F 전송과 기존 테스트 서버 실행은 이번에 명시적으로 허용되었다. 추가 공개 배포 허용으로 해석하지 않는다.

대상은 기존 AMD EPYC7551 물리1코어/논리2CPU·RAM1000.55MB 서버다. CPU affinity0, ORT/BLAS1스레드, sequential, nice19. 다른 서비스는 종료하거나 수정하지 않는다. 워밍업 후 전체 파형 생성 wall time을 측정하고 로딩/파일 저장은 제외한다. RTF=생성시간/오디오길이이며 1 미만이면 실시간보다 빠르다. 공유 서버라 시간 변동이 있다.

메모리는 50ms 표본 감시이며 cgroup 하드 제한이 아니다. 기존 비교는 RSS420MB, 새 엔진 비교는 양쪽 모두480MB와 system available80MB 여유를 적용한다. 실제 최고 RSS·최저 available도 함께 기록한다. 모델 두 개를 동시에 올리지 않는다. ARM 속도는 이번 서버 결과에서 추정하지 않는다.

## E/F 실제 서버 비교

`results/round6_paired_E_F.json`: 순서를 바꾼 fresh process 3쌍, 각 process가 자기 warm-up과 각 문장 측정1회를 수행한다. 두 후보 구조는 같으며 vector 파일48,903,928bytes다.

| 고정 후보 | 한국어 짧은 문장 | 영어 짧은 문장 |
| --- | ---: | ---: |
| E400 | 5.619초 / RTF1.0562 | 5.272초 / RTF1.0685 |
| F1200 | 5.594초 / RTF1.0515 | 5.280초 / RTF1.0702 |

쌍별 속도비 중앙값은 KO1.0044, EN0.9984로 우열이 없다. 최대 표본RSS418.49MB, 최저available235.19MB. 느린 첫 실행/공유 부하 표본도 원시 기록에서 삭제하지 않았다. 이후 엔진 비교는 최종1200회 체크포인트 F로 고정하며 E보다 음질이 낫다고 주장하지 않는다.

F source vector SHA256: `7c81ab7f0ddcc4266743701599a8708d2edd1be04285770f996e84a8bb5f4447`.

## 프로파일과 실험

`results/round6_epyc_operator_costs.json`은 실제 서버 warm+측정1회 profile 합산이다. profiling overhead가 있어 RTF 측정으로 사용하지 않는다. 벡터 추정기 비용의85.18%가 DynamicQuantizeMatMul이며, 보코더75.66%는 FP32 MatMul이다. LayerNorm은 각각0.38%/0.16%라 우선순위를 낮췄다.

1. NumPy/OpenBLAS FP32 행렬곱 교체 probe: ORT MLAS 대비 대략0.78~0.95배라 채택하지 않음.
2. 행64/128개 타일링 probe: 일부 큰 행렬에서 개선하지만 일관된 이득 없음. 전체 그래프 타일링을 구현/채택하지 않음.
3. 정확한 U8×S8 AVX2 정수 커널: 기존 ORT DynamicQuantizeLinear 결과와 기존 S8 가중치를 그대로 사용. [-64,64] 범위를 확인해 PMADDUBSW의 16bit saturation을 피하고 최종S32는 동일하게 계산. 상수 가중치 packing을 세션 내 재사용. 활성값/가중치 재양자화·GELU 근사·스텝 생략 없음.
4. 위 커널과 문장 조건 캐시 결합: 반복부1510→1316노드, 불변194노드·19개 경계텐서를 문장당1회 계산. 결과 음성 자체를 캐시하는 방식은 아님.

커널은 Linux x86_64 AVX2·EPYC용이다. 다른 CPU에는 기본 ORT 엔진을 사용한다. 모든 기본 Engine 설정은 기존대로 유지하고 manifest가 지정하는 실험 모델에서만 확장 커널을 등록한다.

`results/round6_weight_identity.json`은 원본628개 initializer가 변환된 graph/condition의 복사본과 바이트 단위로 같음을 확인한다. graph 직렬화 해시는 달라져도 파라미터는 고정이다. 정수 standalone8형상 및 ORT custom-op ABI4형상/제로포인트/반복 캐시 검사를 수행했다. 실제 짧은 KO/EN PCM16도 기본 F 엔진과 새 커널, 캐시 결합 모두 완전히 일치했다. 긴 문장/다른 음색 확인과 반복 속도 측정은 아래 최종 집계에 추가한다.


## 엔진 v1 검증 완료

핵심 변경은56개 FFN 정수 행렬곱의 정확한 native 커널, 문장 조건 캐시, memory pattern 비활성화다. CPU arena와 prepacking은 유지한다. 패턴을 켠 첫 연속 측정은 긴 영어 실행 중 표본RSS483.10MB로 자체480MB 감시선을 넘었다. 시험 process만 종료됐으며 당시 시스템available191.08MB였다. 이 설정은 채택하지 않았다. benchmark는 이후 case별 부분 JSON과 complete 플래그를 저장하도록 보완했다.

수정 후 짧은 KO/EN→긴 KO/EN 연속 배치가 모두 완료됐다. 각 문장 warm-up 후3회 중앙값이다. baseline배치 다음 engine배치라 공유 부하 차이가 있으며, 아래를 독립 반복 실험 전체의 확정적인 속도비로 해석하지 않는다.

| 문장 | 기존 F1200 시간 / RTF | 엔진 v1 시간 / RTF |
| --- | ---: | ---: |
| KO 짧음 | 5.792초 /1.0888 | 5.504초 /1.0345 |
| EN 짧음 | 5.190초 /1.0519 | 5.036초 /1.0207 |
| KO 김 | 18.787초 /1.0665 | 18.204초 /1.0334 |
| EN 김 | 18.100초 /1.0704 | 17.913초 /1.0594 |

최대 표본RSS는 baseline456.62→engine426.93MB, 최소available201.85→229.38MB였다. 별도 short fresh-process 교차3쌍에서는 baseline KO5.628/EN5.175초, engine5.502/5.090초이며 쌍별 속도비 중앙값은1.0162/1.0166배였다. 이 단계 개선은 작고 실행 변동이 있다. 이전 pattern-enabled short3쌍의1.029배와 설정이 다르므로 합쳐 과장하지 않는다.

M1의4문장(짧고 긴 KO/EN) 및 F1의짧은 KO/EN, 총6개 실제 서버 float32 WAV가 기본F와 샘플 단위로 완전히 같았다. 각 RMSE와 최대차이는0이다. `results/round6_engine_wave_identity_M1.json`, `..._F1.json`에 float byte hash까지 남겼다. 새로운 WER 시험은 하지 않았고, 출력 동일성을 원본교사와의 지각 동등성으로 바꾸어 말하지 않는다.

행/열 loop 순서 변경과4×16 tile도 시험했다. 대부분 개선이 없거나 일부 큰 행렬이 느려 채택하지 않았다. 모든 raw probe를 보존하며 한 번의 큰 outlier만으로 커널을 선택하지 않는다.

### v1 재개 지점

- 기준: `models/round5-original50-student`, 서버 baseline 별칭 `models/round6-F1200`.
- graph 변환: `experiments/exact_integer_backend.py --cache-conditioning`; 실험 패키지 `models/round6-F-exact-cached`.
- build 및 arithmetic ABI 검사: `native/build_exact.py`, `experiments/check_exact_integer.py`.
- 파형 비교: `experiments/check_engine_waves.py`; 수치 집계: `experiments/summarize_round6.py`.
- 표준모델/초기·최종 checkpoint와 original50 data는 [F1200 공개 기록](PUBLICATION_F1200.md)에 고정 revision을 남겼다. 공개 weights를 바꾸지 않고 native 패키지는 내려받은F에서 재생성한다.
- 다음 실험은 정수16bit 부분합이 수학적으로 안전한 block에서만 widening을 합치고, 나머지는 기존 정확한 경로로 처리하는 방법이다. 아직 후보이며 실제 TTS 검증 전에 선택하지 않는다.
