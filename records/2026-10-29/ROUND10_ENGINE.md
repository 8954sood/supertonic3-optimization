# 10차: 커밋 후 RTF 추가 최적화

사용자는9차 커밋 후에도 RTF 개선을 이어가도록 요청했다. 엔진v4·정리 기록은 `7704219`, 선택2배속 Hugging Face 패키지와 GitHub 다운로드 연결은 `8219fc0`으로 main에 커밋·푸시했다. HF model revision은 `f1d05bdb691d8769557e19644f3a63089992daf0`이며 기존 파일118개를 보존했다. 공개 패키지21개 전체를 인증 없이 내려받아 KO/EN 실제8회 출력이 선택 후보와 같은 것을 확인했다. 검증용 중복 다운로드는 지웠다.

이후 실험도 F1200 학습 가중치·양자화·내부1.35/출력2.0·한 스레드·실제8회를 고정한다. 모델/파형 근사로 속도를 얻지 않는다. 기존 EPYC7551 1코어/1GB·affinity0·nice19·480MB RSS/reserve80MB guard를 사용한다. 실제 전체 TTS와 파형 보존을 통과하기 전에는 선택 엔진을 바꾸지 않는다.

## N16 연속 배치 시제품 — 미채택

`experiments/exact_int8_panel16.c`는 인접 N8 weight panel2개를 각 K8 내부에서 연속 배치한다. 기존2개의 안전 플래그는 OR하며, unsafe일 때는 기존 S32 경로를 사용한다. trained tensor를 바꾸는 것이 아니라 세션 내 계산용 정수 packing 순서만 바꾸는 실험이다. N16 이외/K12는 이전 포맷, 행 tail은 새 포맷을 별도로 처리한다.

`experiments/check_exact_integer.py --library <versioned.so>`를 추가해 active library 교체 없이 후보를 검사할 수 있게 했다. 작은3행×16열 tail과 N24 fallback을 추가해10개 ORT S32 differential case가 통과했다. 실제 고정 가중치의8개 shape probe도 정수 결과는 같았다. 하지만 v4 대비 speedup0.983~1.023으로 혼재하여 전체 TTS 후보로 승격하지 않았다. microprobe를 전체 RTF 개선으로 주장하지 않는다.

## Packed weight 버퍼 정렬 — 시제품

v4의 packing 순서·수학을 그대로 두고 계산용 weight 버퍼만64byte 경계에 할당하는 시제품을 만든다. standalone scratch는 매 prepare마다 전체를 다시 pack하므로 grow 시 옛 내용을 복사하지 않아도 되며 실패 시 기존 pointer를 보존하고 prepared state를 무효화한다. ORT node cache도 같은 정렬 할당을 사용한다. round-up overflow를 검사하고 free와 호환되는 C11 aligned_alloc을 쓴다. 추가 학습·새 모델 파일·배속 변경은 없다.

후보 소스는 `experiments/exact_int8_aligned.c`이다. 정수 ABI·실제 가중치 probe를 먼저 확인한다. 선택 엔진은 아직v4이며 추가 gain이 실제로 확인돼야 변경한다.

## 정렬 후보 전체 TTS: 미채택

정렬 후보도10개 정수 ABI와 실제 가중치8개 probe는 S32가 같았다. 그러나 v4 대비2쌍 전체 TTS의 KO speedup은0.998/0.973, EN0.996/1.018로 일관된 이득이 없었다.4개 파형은 모두 같았고 드라이버가 원래v4 library를 복구했다. 정렬이 항상 빨라진다는 가정으로 채택하지 않는다.

## BiasGelu를 보존하는 epilogue: 진행

9차의 전체 epilogue는 bias add까지 제거해 ORT BiasGelu 융합을 바꿀 수 있었다. 이번에는 FFN1의 Cast+Scale만 native로 합치고 bias add는 남기며, FFN2는 Cast+Scale+Bias를 합치는 대조군을 만든다. FP32 cast/multiply/add의 반올림 순서를 유지하고 `-ffp-contract=off`로 빌드한다. 기존 학습 initializer는 그대로다.

첫 준비 검사는 FFN1을 output width>512로 판정했다가28개 count assertion에서 중단됐다. F1200에는 FFN1 width512인 작은2개 블록도 있기 때문이다. 모델/서버 active library를 변경하지 않은 상태에서 걸러졌고, `/pwconv1/` node 구조로 구분하도록 수정했다. 현재 F1200의 고유 FFN 행렬 shape는7종(512×512,512×1024,1024×512,512×1536,1536×512,512×2048,2048×512)이다. 이전8개 microprobe는 큰4종×2개M을 본 대표 표본이며 모든 shape를 망라한 검사가 아니었다. 재현 probe는 이제 고유7종 모두를 순회한다.

Mixed 후보의 정수10개·FP32 scale-only9개·scale+bias9개 oracle 검사를 먼저 수행했고 준비된 전체 그래프의 발음/시간을 별도 검사한다. 순수 standalone 산술 일치를 전체 TTS 개선으로 주장하지 않는다.

## Mixed epilogue 전체 TTS 재검증: 미채택

처음2쌍에서는 KO speedup1.032/1.005, EN1.363/0.972였다. 첫 EN reference만4.679초로 크게 느려져 첫 값은 개선 근거로 쓰지 않았다. 독립적으로 새3쌍을 교대 측정한 결과 KO0.883/0.878/0.972, EN1.005/0.925/1.004였다. KO는 모든 쌍에서 느려져 미채택이다. 총10개 float32 파형은 bit-identical이며 두 드라이버 모두 active v4 복구를 확인했다. 초기 유리한 표본만 고르면 잘못된 결론이 되므로 최초/재검증 원시 결과를 모두 남긴다.

## M3N16 계산 타일: 미채택

`experiments/exact_int8_m3n16.c`는 v4의 M4N16을 M3N16으로 줄인다. 한 번에 필요한 누산 레지스터를8개에서6개로 줄여 compiler의 register pressure를 줄이려는 가설이다. 대신 동일 weight를 재사용하는 행 수가 감소하므로 실제로는 느려질 수 있다. packing·safe-K8 조건·S32 fallback·출력 반올림은 그대로다. 정수 ABI10개와 실제7종 weight shape×M60/200의14개 probe를 먼저 검사한다. 전체 TTS 검증 전에는 채택하지 않는다.

M3N16은10개 정수 ABI와14개 probe에서 결과가 같았다. v4 대비 micro speedup은0.886~1.008, 중앙값0.994로 유리하지 않아 전체 TTS 시험으로 확대하지 않았다.

## 32행 캐시 블록: 미채택

`experiments/exact_int8_block32.c`는 M4N16 산술을 유지하고 순회 순서를32행 묶음 →16열 →4행으로 바꾼다. 계산용 weight panel이 여러4행 타일에 재사용되도록 하려는 가설이다. K 누산 순서·packing·반올림·모델은 그대로다. 기존6차의 단순 col-first 결과와는 구분하며 실제 v4·F1200에서 검증한다. 빌드 소스·library 해시와 상대 경로 재현 명령은 `results/round10_experimental_builds.json`에 보존한다.

32행 블록도10개 정수 ABI와14개 probe에서 S32가 같았지만 micro speedup0.961~1.012, 중앙값1.000으로 개선이 없어 미채택이다.

## 종료 상태와 다음 작업

후속5개 후보는 모두 미채택이며 선택 엔진은v4다. 전체 TTS를 실행한 정렬/혼합/혼합 재검증의14쌍 float32 파형이 모두 같고, active library SHA256도 원래v4와 같다. 이 결과는 추가 음질 개선이나 새 WER/CER/MOS 측정이 아니다. 실제 서버의 최신 채택 RTF는9차의 정배속 KO1.0026/EN1.0036,2배속 KO1.3506/EN1.3542이며 아직2배속 실시간 생성에는 미달한다.

성공한v4와 선택2배속 모델 공개는 이미 main에 커밋·푸시했다.10차에는 새 속도 개선이 없어 추가 커밋/푸시나 HF 재업로드를 하지 않았다. 실패 시제품과 원시 결과·검사 도구 확장은 로컬 작업으로 남긴다. 다음 작업은 `round10_state.json`과 세 개의 paired 결과부터 읽고, 이 다섯 가설을 검증 없이 반복하지 말아야 한다. 첫 혼합 후보의 EN1.363배나9차 KO의 외부 편차를 전체 개선 수치로 쓰지 않는다.

빌드는 저장소 루트에서 `results/round10_experimental_builds.json`의 각 command로 재현한다. mixed만 `-ffp-contract=off`가 필요하다. graph 생성은 `experiments/fuse_exact_epilogue.py --source models/round8-F-retime --output models/round10-F-retime-mixed --preserve-bias-gelu`이며 원본56 FFN과28 retained bias assertion 및 initializer byte 비교를 통과해야 한다. CPU library 파일은 Linux x86_64 AVX2용이며 Mac/ARM 결과로 일반화하지 않는다.10차는 새 학습 모델이나 새 dataset을 만들지 않았다.

## 공개 기록 반영

사용자가 후속 지시에서10차 실험도 커밋하도록 명시했다. 기존 미채택 결론을 유지한 채 시제품5종·검사 도구·원시 측정·인계를 main에 보존한다. 선택 엔진은v4이며 새로운 속도 개선이나 모델 릴리스는 없다. 다음 작업은 Mac 로컬의 스트리밍 서버 구현·평가다.
