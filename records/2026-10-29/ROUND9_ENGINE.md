# 9차: 내부 배속 후보의 추론 엔진 최적화

## 조건

사용자는 선택 후보의 정상·느린·빠른 음성을 F1200과 비교한 뒤 RTF 개선을 요청했다. F1200 가중치·양자화·내부1.35/출력2.0·실제8스텝을 고정한다. 파형 후처리·모델 축소·다른 학생 교사 사용은 하지 않는다. 기존 EPYC7551 1코어/1GB 서버에서 nice19, affinity0, ORT/BLAS1스레드로 측정한다. 성공한 최적화만 main에 커밋·푸시한다.

## 청취 묶음과 정상 속도

로컬 `samples/round9_speed_listening`에 M1/F1 ×0.7/1.0/1.5/2.0 × F1200/선택 후보16개를 생성해 A/B 비교 페이지·ZIP으로 제공했다. seed42·실제8회·같은 문장이다.0.7/1.0/1.5의6쌍은 float32 파형이 완전히 같다.2.0의2쌍만 다르다. 현재 개선 경로는2.0에만 적용된다. 이 샘플은 품질 청취용이며 속도 측정으로 해석하지 않는다.

RTF는 생성시간/최종음성길이이다. 출력2.0이라 길이는 정상의 절반이지만 내부1.35에서8회 계산하므로 계산량이 절반까지 줄지 않는다.8차의RTF1.39~1.45를1배속 자체가 그만큼 느려졌다는 뜻으로 해석하지 않는다. 실제 정상 속도 비용도 별도 재측정한다.

## 후보 실험

v2의 K8 안전 부분합을 유지하면서 연속 K32/K64 그룹의 플래그가 모두 안전할 때만 내부 조건 분기를 생략하는 시제품을 만든다. 각 K8 뒤의 S32 widening은 그대로이므로 K16 범위를 가정했던7차 실험과 다르다. 그룹 내 unsafe가 하나라도 있으면 원래 K8 안전 검사/정확한 fallback을 수행한다. 가중치 packing 포맷과 정수 산술은 유지한다. 아직 선택하지 않았으며 실제 가중치 microprobe → 정수 ABI/경계 → 전체 파형 → 교대 서버 측정 순서로 검증한다.

## 긴급 디스크 정리 — 사용자 명시 요청

사용자가 예전 불필요 가중치 대부분의 삭제를 요청했다. 로컬 디스크 여유는152,715,264bytes(약152MB)였다. 실패한/오래된 모델 ONNX·ORT·중간 checkpoint와 검증용 중복 다운로드 가중치 및 오래된 서버 패키지1개를 삭제했다.640개 파일의 논리 크기는37,722,414,274bytes이며, 실제 available은34,155,810,816bytes(약34.16GB/31.81GiB)로 늘었다. hardlink/APFS 때문에 논리 파일 합계와 실제 회수량은 다르다.

보존한 모델 폴더: 원본 `supertonic-3`, F1200 `round5-original50-student`, 선택한 `round8-latent-retime-g135`, 엔진v2 `round6-F-exact-cached`, 재현용 `cfg-student-fp32-source`와 `fast-fp32-cached`, 평가기 `whisper-base`. 학습 checkpoint는 F1200 best와 원본50 초기화 initial을 보존했다. 새 엔진 시험은 기존 원격 패키지에서 진행하므로 영향을 받지 않았다.

소스·manifest·실험 JSON·학습 데이터·비교 WAV와 페이지는 유지했다. 실패한 학습 가중치들은 더 이상 로컬에 없으며, 과거 manifest가 폴더에 남아 있어도 실행 가능한 모델이라는 뜻이 아니다. 필요하면 공개된 고정 HF 리비전에서 새 폴더로 내려받거나 보존한 코드/원본으로 재생성한다. 삭제 목록·크기·보존 목록은 `results/round9_weight_cleanup.json`에 기록했다. 다른 작업 폴더나 시스템/개인 파일은 정리 대상으로 삼지 않았다.

## 초기 엔진 진단

원격1배속 재측정은 선택 후보 KO5.847/EN5.407초, 기존 엔진v2 F1200은5.845/5.548초였다(각 문장 warm-up 뒤2회 중앙값, 서로 다른 fresh process). 두 경로의1배속 비용은 비슷하며 영어 수치 차이를 후보의 가속으로 주장하지 않는다.8차의RTF 상승은 주로2배속 출력 길이 대비 내부 생성 계산량에서 비롯한다.

선택 후보2배속의 profile에서 vector 시간의69.29%가 native exact integer GEMM,8.67%가 나머지 DynamicQuantizeMatMul이었다. 보코더는75.52%가 FP32 MatMul이다. profile은 overhead가 있으므로 최종 속도 비교에 쓰지 않는다(`round9_epyc_operator_costs.json`).

K32/K64 안전 플래그 그룹 시제품은 실제8개 가중치/shape에서 정수값은 같았으나 대체로 느려 채택하지 않았다. K32 speedup범위0.931~1.008, K64는0.921~1.017이다. 반면 K8 안전 부분합을 유지한4행×16열 tile은1.015~1.119의 microprobe 개선을 보였다.6차의4×16은 K8 부분합이 없는v1 비교였으므로 이번v2 실험과 구분한다. 아직 전체 TTS 검증 전이다.

또한 기존 정수 결과의 FP32 Cast→Mul→Add를 같은 custom operator에서 처리하는 후보를 만들었다.56개 FFN에서168개 중간 노드를 제거하며 initializer는 전부 바이트 단위로 보존한다. 원래 FP32 scale product는 그래프에 남기고 cast, multiply, add 순서와 반올림을 유지한다. 컴파일은 `-ffp-contract=off`로 FMA 결합을 막는다.9개 standalone ORT/정수-oracle case는 bit-identical이지만 전체 그래프의 최적화 조합과 음성도 별도 확인해야 한다. tile 단독·epilogue 단독·결합을 순차 fresh-process로 시험하고 종료 시 원래 active library를 원자적으로 복원한다.

## 전체 TTS 교대 비교 — tile 선택, epilogue 미채택

내부1.35/출력2.0·M1·KO/EN short, 각 fresh process는 warm-up1회 뒤1회 측정한다. 순서를 뒤집는3쌍 비교에서 v2→tile의 쌍별 speedup은 KO1.289/1.042/1.035, EN1.040/1.063/1.047이었다. 중앙값은 각각1.0418/1.0468이다.6개 비교 WAV가 모두 float32 bit-identical이다. 별도 시간 중앙값은KO3.979→3.772초, EN3.679→3.508초이며 후보 RTF는1.3506/1.3542다. 첫 KO reference 지연을 숨기지 않고 raw report에 남겼다. 이 outlier를 포함한 평균 가속비로 과장하지 않는다(`round9_paired_tile_speed2.json`).

Epilogue 단독/결합 시제품도 짧은2개 파형은 모두 같았다. 그러나 초기 비교의 KO에서는 공통 보코더 시간까지 크게 늘어 공유 환경 변동이 섞였다. 이를 특정 후보의 연산 비용으로 단정하지 않았다. tile 대비 combined를 순서 반전2쌍으로 재검사했다. KO speedup1.007/0.998, EN0.997/0.980으로 일관된 추가 이득이 없어 **epilogue 그래프/커널을 채택하지 않았다**. 선택은4×16 safe-K8 tile 단독이며 모델 그래프도 기존8차 그대로다(`round9_paired_tile_combined_speed2.json`).

선택 커널을 엔진v4로 구분한다.7차의 K16 v3는 계속 미채택 상태다. `native/exact_int8.c`와 `native/build_exact.py`가 v4를 준비한다. 새 빌드의 ELF .text/.rodata/.data/.eh_frame은 시험한 tile binary와 모두 같으며 소스 파일명 등에 따른 전체 파일 hash 차이는 별도 기록했다(`round9_v4_build.json`). 최종 ABI·정상·장문·여성 출력 검증을 이어간다.

최종 검사의 첫 실행은 원격 `speed_sweep.py`에7차 보류 실험의 `speed_mode` 인자가 남아 현재 runtime과 맞지 않아 생성 전에 중단됐다. 메모리 감시 중단이나 모델 품질 실패가 아니다. 실패 report를 별도 이름으로 보존하고 old v2 library를 finally에서 복구했다. 원격 검사 스크립트를 로컬의 표준 native 인터페이스와 맞춰 재실행한다. 실패한 결과는 성공 수치에 섞지 않는다.

## 최종 검증과 적용

원격 검사를 다시 실행해8개 정수 ABI case(±32 안전 경계·+64 fallback·K12·여러 batch/row tail·zero0/255·반복 cache)를 통과했다. 정상M1 KO/EN,2배속M1 KO/EN 장문, F1 KO/EN 정상/2배속 총8개 WAV가 각각 기존 선택 후보와 float32 샘플까지 완전히 같았다.3쌍 비교의6개 fast-short 출력 검사를 포함하면14개 A/B 검사가 모두 통과한 것이다. 원본50 교사와의 동일성이나 새 ASR 품질 개선을 주장하지 않는다.8차의 잔여 한국어 발음 오류는 그대로다.

모든 성공 guard는480MB/reserve80MB 안에서 완료됐다. 실패하면 자동으로v2를 복구하는 검증 드라이버가 전체 통과 후에만 원격 active library를v4로 선택했다. 모델·배속·step 설정은 바꾸지 않았다. 새 Hugging Face 모델/데이터 업로드도 하지 않았다. 로컬 원본 공식 실행기와 비교한 기존 `tests_runtime.py`2개 검사도 통과했다. 디스크 정리 후 선택 F1200 vector와 retime vocoder의 SHA256을 다시 검사해 보존을 확인했다.

재현 시 기준v2 소스는 이전 main `63e8672`의 `native/exact_int8.c`에서 얻고, 새v4는 현재 `native/build_exact.py`로 별도 이름에 빌드한다. 두 binary를 동시에 사용하는 프로세스를 띄우지 말고 아래 드라이버로 순차 비교한다. 드라이버는 library swap을 자식 프로세스 사이에서만 수행하고 종료 시 원래 파일을 복구한다.

```bash
python native/build_exact.py --output native/libexact_v4.so
python experiments/check_exact_integer.py --output results/v4-abi.json
python experiments/probe_exact_tiles.py \
  --model models/F1200-retime-exact \
  --reference-library native/libexact_v2.so --candidate-library native/libexact_v4.so \
  --output results/v4-probe.json
python experiments/paired_native_kernels.py \
  --reference-library native/libexact_v2.so --candidate-library native/libexact_v4.so \
  --model models/F1200-retime-exact --speed 2 --pairs 3 \
  --output results/v4-paired.json
```

`check_exact_integer.py`는 현재 active `native/libexact_int8.so`를 검사하므로 먼저 검증할 binary를 해당 경로에 설치한다. probe는 packing이 끝난 정수 연산만 측정하고 전체 TTS RTF로 해석하지 않는다. 전체 비교에는 warm-up·가중치 로딩 제외 범위·쌍별 수치·파형 identity가 포함된다. 정상 속도는 `--speed 1`, 여성/배속 기능은 `experiments/speed_sweep.py --voice F1 --speeds 1 2`, 장문은 `benchmark.py --cases ko_long en_long --speed 2`로 따로 검사한다. 배속별 reference/candidate 폴더는 `experiments/check_engine_waves.py`로 비교한다.

K32/K64와epilogue/combined는 `experiments/exact_int8_{grouped,epilogue,combined}.c`에 시제품을 보존한다. grouped는 `-DEXACT_KGROUP=32` 또는64, epilogue/combined는 반드시 `-ffp-contract=off`로 빌드한다. 공통 빌드 옵션은 Linux x86 target `x86_64-linux-gnu.2.28`, CPU`znver1`, `-O3 -Wall -Wextra -Werror -fPIC -shared -I native/include`이다. 이 실패 후보는 기본 엔진으로 쓰지 않는다.

최종 별도 warm 측정: 정상1.0 KO5.600초/RTF1.00257, EN5.199초/RTF1.00357(각2회 중앙값).2배속 장문은KO12.313초/RTF1.33138, EN11.720초/RTF1.32019(각1회). 별도 batch와 이전 batch를 비교해 쌍별 가속으로 주장하지 않는다. 최종 검증 전체의 표본 최대process-tree RSS는394.58MB였다.1배속은약RTF1.00,2배속은약1.32~1.35이며 지속 실시간 보장은 아니다. 정상 파형이 같다는 사실과2배속 계산량의 추가 비용을 구분한다.

최신 기계 판독 상태는 `python experiments/summarize_round9.py`로 원시 JSON에서 `round9_state.json`을 재생성한다. 사용자가 추가로 HF 공개를 요청했으므로 별도 공개 기록에서 패키지·고정 리비전을 연결한다. 이번 엔진은 학습 가중치를 바꾸지 않으므로 새 훈련 checkpoint는 없다.
