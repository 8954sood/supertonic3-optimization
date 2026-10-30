# 17차: Linux x86 v5 — 정확한 연산 유지와 제한적인 RTF 개선

선택 기준은 main `7db1daa`의 F1200·엔진 v4다. 사용자는 실제 1코어·1GB 서버에서 성능을 개선하는 v5를 요청했다. **최종 선택은 lossless sparse correction + M2/M1 safe-K8 tail + 배치 1의 v5 조합**이다. 작은 행렬 시험의 가속비를 전체 TTS 개선으로 바꾸어 발표하지 않는다. 정배속 KO는 사실상 같으며 가장 강한 근거는 M1 EN 2×의 약 1.99% 개선이다. 실패 후보와 확대 검사의 불리한 실행도 모두 보존한다.

## 고정 조건

16차와 같은 EPYC 7551, 논리 CPU 0 하나, 수치 라이브러리 및 ORT intra/inter 1/1, nice19다. 원래 서버 파일은 덮어쓰지 않는다. 별도 `round16-eval`에 최신 runtime과 버전별 라이브러리를 두고 각 세션에 명시적으로 등록한다. F1200 가중치·양자화·실제 VE 8회·기존 배속 처리는 고정했다. ARM 정규화는 사용하지 않는다.

50ms 표본 guard는 자기 자식 프로세스 트리 RSS 480MB, 시스템 available 80MB 및 작업별 시간 상한을 확인한다. 이는 OS 강제 제한이 아니다. 전체 TTS 비교는 float32 파형 SHA·길이·실제 호출 수를 확인한다. 새 WER/CER/MOS 측정이나 모델 학습·HF 업로드는 없다.

## 정수 커널의 가설과 시험

- `spans`: N16 패널별 unsafe K8 위치를 미리 저장하고 안전 구간을 연속 실행한다.
- `likely`, `likely_unroll`: 안전 분기 힌트와 K8 루프 2회 전개를 시험한다.
- `gcc`: 같은 v4 C를 GCC 13.3으로 컴파일한다. 서버 시스템 패키지는 설치·업그레이드하지 않고 별도 폴더에 패키지를 풀어 assembly를 생성한 뒤 개발 Mac의 Zig로 링크했다. 패키지·소스·assembly 해시는 build 기록에 있다.
- `memones`: int16→int32 합산에 필요한 상수 벡터를 메모리 피연산자로 바꿔 SIMD 레지스터 하나를 확보한다. `memones_ordered`는 갱신 순서 제약도 추가한다. `selective`는 초기 shape 시험에서 유리했던 크기에만 적용한다.
- `adaptive`, `adaptive_m3`: signed 범위 증명으로 길이가 다른 int16 부분합 그룹을 만든다. 각각 M2/M3 타일이다.
- 정수의 모든 부분합이 FP32 정수 정확 범위 안인지 증명하고 FP32 BLAS도 시험했다. 결과는 같았으나 약 2배 이상 느려 미채택이다.
- `specialized`: 실제 7종 K/N 크기를 컴파일 때 상수화한다.
- `joined_flags`: 두 N8 안전 플래그의 OR를 packing 때 처리한다.
- `xmm`: Zen1의 128비트 폭에 맞춘 M4N8 연산을 시험한다.
- `safe_panels`: 전체 K 구간이 안전한 N16 패널에만 별도의 분기 없는 루프를 쓴다.
- `sparse_correction`: unsafe K8의 뒤쪽 K4 항을 사설 packed 표현에서 분리한다. 기본 루프는 항상 안전한 K8을 계산하고 분리한 항을 S32로 모두 더한다. 원래 열 합계와 zero-point 보정도 유지한다. 근사나 가중치 삭제가 아니다. dense 최악 용량은 예약하지만 실제 sparse 레코드만 쓰며 RSS는 별도로 감시한다.
- `tails`: 원래 M4 루프 뒤 1~3행이 K4 단일 행 fallback을 쓰던 부분에 M2/M1 N16 safe-K8 경로를 추가한다. `sparse_tails`, `sparse_tails_memones`는 위 가설과의 조합이다.
- `m10n8`, `m12n8`: 메모리 피연산자 widening으로 레지스터를 확보하고 가중치를 재사용하는 행 수를 늘린다. 남는 행은 기존 v4로 보낸다.

각 build JSON은 소스/바이너리 SHA와 상대 경로 명령을 보존한다. `experiments/check_exact_integer.py`는 표준 ORT S32 결과와 대조한다. 12개에서 시작해 M2 잔여 행·M10/M12의 dense unsafe와 N8 경계를 추가했으므로 파일별 실제 case 수를 읽어야 한다. 기존 결과를 새 case 수로 소급하지 않는다.

아래는 실제 고정 FFN 가중치의 작은 시험이다. 기본은 7종 shape × M60/M200, tail 조합은 M56/M60/M75/M81을 사용했다. 현재 기본 시험은 같은 shape의 첫 가중치만 고르므로 전체 56개를 대표한다고 가정하지 않는다. 후속 도구에 `--all-weights`를 추가했다. 가속비 1보다 커야 빠르다.

| 커널 | case 수 | 가속비 중앙값 | 빨랐던 case | 범위 |
| --- | ---: | ---: | ---: | --- |
| [spans](../../results/round17_spans_probe.json) | 14 | 0.9492 | 1 | 0.8874–1.0460 |
| [likely](../../results/round17_likely_probe.json) | 14 | 0.8540 | 0 | 0.5570–0.9078 |
| [likely_unroll](../../results/round17_likely_unroll_probe.json) | 14 | 0.9085 | 0 | 0.8349–0.9356 |
| [gcc](../../results/round17_gcc_probe.json) | 14 | 0.9737 | 0 | 0.9341–0.9960 |
| [memones](../../results/round17_memones_probe.json) | 14 | 1.0078 | 10 | 0.9816–1.0245 |
| [memones_ordered](../../results/round17_memones_ordered_probe.json) | 14 | 0.9987 | 7 | 0.9375–1.0376 |
| [adaptive](../../results/round17_adaptive_probe.json) | 14 | 0.8612 | 0 | 0.8252–0.9838 |
| [adaptive_m3](../../results/round17_adaptive_m3_probe.json) | 14 | 0.8890 | 1 | 0.8210–1.0408 |
| [selective](../../results/round17_selective_probe.json) | 14 | 1.0129 | 11 | 0.9438–1.1073 |
| [specialized](../../results/round17_specialized_probe.json) | 14 | 1.0031 | 8 | 0.9648–1.0561 |
| [joined_flags](../../results/round17_joined_flags_probe.json) | 14 | 0.9937 | 4 | 0.8763–1.0162 |
| [xmm](../../results/round17_xmm_probe.json) | 14 | 0.9892 | 3 | 0.9662–1.0282 |
| [safe_panels](../../results/round17_safe_panels_probe.json) | 14 | 0.9958 | 4 | 0.9665–1.0475 |
| [sparse_correction](../../results/round17_sparse_correction_probe.json) | 14 | 1.0101 | 9 | 0.9408–1.0545 |
| [tails](../../results/round17_tails_probe.json) | 28 | 1.0035 | 19 | 0.9424–1.0557 |
| [sparse_tails](../../results/round17_sparse_tails_probe.json) | 28 | 1.0094 | 20 | 0.9781–1.0445 |
| [m10n8](../../results/round17_m10n8_probe.json) | 14 | 0.8599 | 0 | 0.8264–0.8814 |
| [m12n8](../../results/round17_m12n8_probe.json) | 14 | 0.8563 | 0 | 0.7941–0.9351 |
| [sparse_tails_memones](../../results/round17_sparse_tails_memones_probe.json) | 14 | 1.0080 | 11 | 0.9798–1.0354 |

## 전체 TTS와 서버 변동

`paired_server_engines.py`로 v4 / 추가 정수 linear 56→84 / 추가 linear+선택적 memones를 세 쌍씩 새 프로세스에서 교대했다. 18개 프로세스가 완료됐고 파형은 같았다. 그러나 추가 linear의 KO 2×, 결합의 EN 2×에서 크게 느린 실행이 나왔다. 이 결과를 버리지 않았고 채택하지 않았다(`round17_matrix.json`).

이어 `paired_shared_frontend.py`를 사용했다. frontend·condition·vocoder·음색 파일과 배속 설정의 SHA/동일성을 확인한 뒤 한 프로세스에서 VE 세션 두 개만 번갈아 실행한다. 같은 전처리·보코더를 공유해 프로세스 로딩과 장시간 순서 편차를 줄인다. 두 VE의 추가 상주 메모리는 시험 비용이며 제품 RSS가 아니다. warm-up은 별도로 하고 모든 비교에서 실제 VE 8회와 float32 SHA를 검사한다. 각 호출의 process CPU time과 CPU0 steal 누적 차이도 기록하지만 이를 이용해 보정한 RTF를 만들지는 않는다.

변경하지 않은 vocoder에도 큰 지연이 발생했다. 예를 들어 첫 결합 비교 EN 정배속에서 vocoder가 약 1.8초에서 2.5초로 늘어난 실행이 있다. 이것만으로 특정 커널의 인과 효과나 VM의 원인을 단정하지 않는다. 가장 빠른 실행만 선택하거나 임의로 outlier를 제거하지 않는다.

| 공유 세션 비교 | 쌍 수 | KO 1× | EN 1× | KO 2× | EN 2× |
| --- | ---: | ---: | ---: | ---: | ---: |
| [추가 linear + 선택 memones](../../results/round17_shared.json) | 6 | +1.451% | -7.098% | +1.442% | +2.783% |
| [선택 memones 단독](../../results/round17_memones_shared.json) | 6 | +1.050% | -0.683% | +0.173% | +1.744% |
| [lossless sparse correction](../../results/round17_sparse_shared.json) | 4 | -2.992% | +1.019% | +1.044% | +1.264% |
| [배치 1 명시](../../results/round17_batch_one_shared.json) | 4 | +2.267% | +0.659% | +0.802% | -0.158% |

표는 **쌍별 전체 생성 시간 감소율의 중앙값**이며 양수가 개선이다. RTF 자체와 모든 실행은 각 JSON에 남겼다. 쌍별 중앙값은 별도 시간 중앙값끼리의 비율과 다를 수 있다. 일관되지 않은 1% 안팎의 수치를 확정 성능 개선으로 발표하지 않는다.

## 추가 그래프 후보

`prepare_exact_linears.py`는 남은 28개 eligible MatMulInteger를 기존 exact kernel로 연결한다. 기존 56개 custom op를 확인하고 weight 범위·zero point·K/N 조건·직렬화 initializer의 완전 보존을 검사한다. 소스는 읽기 전용으로 유지하고 완성된 패키지만 원자적으로 공개한다. 단위 검사 3개와 서버 파형 대조를 통과했으나 속도 채택은 별도다.

`experiments/freeze_native_batch.py`는 현재 API의 단일 문장 배치 1을 명시한다. noisy_latent의 첫 차원을 읽는 Gather 하나를 정수 상수 1로 바꾼다. 기존 544개 initializer를 그대로 보존하며 새 shape 상수만 1개 추가한다. 고정 key 투영 네 곳은 현재 CPU의 ORT가 세션 로딩 때 계산할 수 있고, Mac에서 계산한 FP 상수를 배포하지 않는다. 기존 condition·모델 설정·가중치·배속은 같다. 준비 도구의 원본/중간 실패 보존 단위 검사 3개를 통과했다. 네 조합의 파형도 v4와 같지만 시간 감소율이 작아 추가 검증이 필요하다.

## 10쌍 재검증 — x86 v5 조합

`models/batch-one` + `libexact_sparse_tails.so`를 기준 v4와 비교했다. 이 조합은 lossless sparse correction, M2/M1 safe-K8 tail, 배치 1의 shape 명시다. 고정 seed 451로 AB/BA 각 5쌍의 순서를 섞었으며 속도별 KO/EN 짧은 문장을 10쌍씩 실행했다. 모든 40쌍의 float32 파형과 실제 VE 8회가 같았다.

| 조합 | v4 RTF 중앙값 | v5 후보 RTF 중앙값 | 쌍별 시간 감소율 중앙값 | 빨랐던 쌍 |
| --- | ---: | ---: | ---: | ---: |
| KO 1× | 1.01646 | 1.01736 | +0.1845% | 6/10 |
| EN 1× | 1.02422 | 1.01195 | +1.0147% | 8/10 |
| KO 2× | 1.36043 | 1.34902 | +0.8912% | 8/10 |
| EN 2× | 1.38286 | 1.35281 | +1.9891% | 10/10 |

정배속 KO는 실질적으로 같다. 쌍별 비율의 중앙값과 별도 RTF 중앙값의 비율은 다르므로 앞 두 RTF 열로 감소율 열을 다시 계산하면 안 된다. outlier를 삭제하지 않았다. 탐색적 bootstrap 20,000회에서 EN 2×의 중앙값 구간은 약 +1.15~+3.04%였지만 다른 세 조합은 0을 포함했다. 이것을 모든 문장·음색의 보장 가속률로 해석하지 않는다. 원시: `round17_combined_confirm.json`, 불확실성: `round17_confirmation_uncertainty.json`.

공유 VE 두 개를 담은 시험의 표본 최대 RSS는 466.20MB, 최소 system available은 186.19MB였다. 이 수치를 단일 제품 프로세스의 RSS로 쓰지 않는다. 여성 음색의 fresh-process 비교와 장문/단일 프로세스 검증을 추가한다.

소스를 `native/exact_int8.c`의 v5 후보로 정리하고, 실패 빌드가 기존 라이브러리를 훼손하지 않도록 빌더를 임시 파일→교체 방식으로 고쳤다. 측정한 sparse_tails 바이너리와 공개 소스 빌드의 .text/.rodata/.data/.eh_frame SHA가 모두 같다. 17개 정수 oracle과 실제 원본 56개 FFN×네 행 크기도 검사했다. 모든 56개를 합산한 kernel 가속비는 M56/60/75/81에서 각각 약 1.0077/1.0040/1.0212/1.0078이며 전체 TTS 가속비가 아니다. 같은 v4를 양쪽에 둔 control은 약 0.9973~1.0032였다.

`prepare_native_batch.py`는 실험 도구를 재현 가능한 준비 도구로 정리한 것이다. 공개 HF retime 모델 → 기존 exact 변환 → 새 준비 도구로 만든 다섯 그래프의 SHA가 실제 측정 패키지와 모두 같았다(`round17_v5_reproduction.json`). 모델 학습이나 새 learned tensor가 없다.

## ARM 추가 조사 — 미채택

같은 정수 곱을 ARM SDOT로 계산하는 M4/M6 후보를 만들었다. 처음 후보는 느렸다. 입력을 한 번 4행씩 묶고 [ARM의 lane SDOT](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)에서 재사용한 M4/M8/M12 후보는 17개 S32 oracle을 통과했다. M8의 14개 실제 weight 작은 시험은 표준 ORT S32 연산보다 약 1.058~1.105배 빨랐다. 이것을 전체 음성 속도 개선으로 발표하지 않는다.

실제 현재 Mac 패키지와 비교하자 파형이 달랐다. 첫 KO 정배속의 latent 상대 RMSE는 첫 step 약 0.004, 마지막 약 0.115였고 wave 상대 RMSE는 약 0.860이었다. shape 정보를 제거한 대조도 같은 결과였다. 이 후보를 채택하지 않았다(`round17_arm_parity_diagnostic.json`).

[ORT의 동적 양자화 구현](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_matmul.cc)과 [backend 선택 설정](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)을 확인했다(2026-10-30). 양쪽 엔진에 `mlas.disable_kleidiai=1`을 적용한 별도 대조에서는 float32 파형이 완전히 같았다. 현재 Mac의 기본 backend 경로와 분리한 S32 경로를 바꾸는 것이 차이의 원인임을 좁혔다. 정확히 어느 양자화/FP 연산에서 최초 차이가 나는지는 추가 trace가 필요하다. 기본 backend를 꺼서 느려진 결과를 개선으로 채택하지 않는다. Oracle ARM N1에서 실행한 검사가 아니다.

별도로 정규화 입력 타일을 잠시 전치해 연속 로드하는 후보도 만들었다. 원래 Welford/FMA 순서는 유지하고 30개 설치 ORT byte 검사를 통과했다. 하지만 실제 네 조합·3쌍에서 한 조합만 빨랐으므로 미채택했다(`round17_norm_transpose_screen.json`). 선택 Mac 커널은 계속 기존 16행이다.

## 확대 검사와 최종 검증

여성 F1의 별도 프로세스 3쌍 비교는 네 조합 모두 파형이 같았으나 KO 1×/EN 2×의 시간 감소율 중앙값이 -2.67%/-5.67%였다. EN 1×/KO 2×는 +1.36%/+1.77%였다. 느린 실행에서 변경하지 않은 전처리·condition·vocoder 시간도 늘었으므로 이 배치만으로 엔진의 회귀나 서버 변동을 단정하지 않았다(`round17_f1_fresh.json`).

불리했던 F1 KO 1×와 EN 2×를 각각 균형 AB/BA·10쌍으로 짧은 간격에 재검사했다(seed 943/944). 감소율 중앙값은 각각 +0.5079%, +0.6756%였다. 빨랐던 쌍은 6/10, 7/10이므로 이 역시 큰 가속이나 보장 성능으로 해석하지 않는다. 불리한 첫 배치를 삭제하거나 대신한 것이 아니다(`round17_f1_ko1_confirm.json`, `round17_f1_en2_confirm.json`).

M1 KO/EN 장문·1×/2× 네 조합은 16차 v4와 float32 SHA·길이·샘플레이트가 같았다. 서로 다른 배치이므로 장문 가속률은 주장하지 않는다. 후보 RTF는 KO/EN 1× 1.0424/1.0130, 2× 1.3326/2.0713이었다. 특히 마지막 EN은 전처리·condition·VE 지연이 함께 커졌으며 그대로 기록한다(`round17_v5_long_checks.json`).

F1·1.25×의 KO/EN도 v4와 같은 float32 파형이었고 각 호출은 실제 8회였다. 선택 조합의 단문 교대 비교는 반복을 포함해 총 **74쌍**, 고유 단문 설정은 **10개 조합**이다. 여기에 장문 4개 설정을 대조했다. 같은 두 단문을 여러 음색/배속으로 반복한 것이므로 74개의 독립 문장을 평가했다고 표현하지 않는다. 새로운 WER/CER/MOS 측정은 없다.

단일 엔진을 담은 v5 검사들의 표본 최대 프로세스 트리 RSS는 약 **438.72MB**였다. 공유 VE 비교의 466MB와 구분한다. 모든 guard가 통과했으며 이 수치는 검사한 입력 범위의 관측값이지 모든 길이의 상한이 아니다.

대상 서버에서 ORT가 최적화한 그래프도 저장·검사했다. 기존 static key projection은 4개였고 v5에서는 0개였으며 exact FFN 56개는 유지됐다. 이 검사는 세션 로딩/그래프 분석이며 추론 속도 측정이 아니다(`round17_target_folding.json`).

검증 요약:

- 공개 v5와 보존한 v4 빌드 모두 17개 S32 oracle 통과. v4 보존 소스 SHA는 `7db1daa`와 같고 기존 frozen 바이너리의 네 계산 section도 같다.
- 공개 v5 소스의 네 ELF section은 실제 M1 10쌍에서 사용한 sparse_tails prototype과 같음. 실패한 빌드가 기존 라이브러리를 보존하고 임시 파일을 제거하는 검사 통과.
- runtime/서버/준비 도구 21개 검사 통과. 복사와 선택적 hard-link 준비를 추가한 뒤 관련 3개를 다시 검사함. 소수점 두 자리 배속·SSE/PCM·취소/queue 검사 포함.
- 공개 HF retime → exact → `prepare_native_batch.py`로 재현한 다섯 그래프가 실제 서버 측정본과 같은 SHA. 최신 도구로도 재확인함.
- 최소 실행 패키지에 빠져 있던 `exact_ops.py`·C·빌더·ORT header를 포함하도록 수정. 바이너리는 포함하지 않음. 추출한 소스의 import/CLI와 Linux cross-build section 일치를 검사했다. Mac에서 Linux 바이너리를 실행한 검사로 표현하지 않는다.

## 배포와 이어서 작업하기

공개 소스는 `native/exact_int8.c`, 원래 대조 소스는 `native/exact_int8_v4.c`다. `native/build_exact.py`의 기본은 v5이며 `--engine-version 4`로 되돌릴 수 있다. 빌드는 임시 파일에서 완료한 뒤 원자적으로 교체한다. 기존 프로세스는 종료하고 새 프로세스로 해당 버전을 실행한다.

준비 도구는 `prepare_native_batch.py`다. 기본은 복사이며 `--link-assets`는 공간 절약을 위해 바뀌지 않은 assets를 공유한다. 링크한 두 패키지는 읽기 전용으로 유지한다. 변환/빌드는 여유 RAM이 있는 개발 기기에서 수행하고 Linux용 `.so`와 완성된 모델만 제한 서버에 옮길 수 있다.

학습 가중치·dataset은 바뀌지 않았으며 기존 공개 HF revision으로 재현한다. 새 learned checkpoint나 dataset 업로드는 필요하지 않다. 최종 개발기기 패키지는 `models/F1200-retime-v5`다. Mac 청취 서버와 ARM 행 병렬 정규화는 기존 15차를 유지한다. 원래 서버 작업 폴더의 소스/모델/라이브러리와 다른 서비스는 이번 격리 시험에서 덮어쓰지 않았다.

후속 최적화는 작은 1% 차이를 과장하지 않고 새 기준과 교대 비교해야 한다. ARM native를 다시 시도할 때는 S32 검사만으로 전체 파형 동등성을 가정하지 말고 기본 KleidiAI 경로까지 대조해야 한다. Oracle ARM 실기기 측정은 별도이며 x86 v5를 ARM에 그대로 적용하지 않는다. 최신 집계와 상태는 `results/round17_summary.json`, `round17_state.json`에 있다.
