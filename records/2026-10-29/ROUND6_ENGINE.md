# 6차: 모델 고정, 실제 1코어·1GB 서버 엔진 최적화

진행 중인 로컬 기록. Git/Hugging Face 공개 업로드·커밋·푸시는 하지 않는다.

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
