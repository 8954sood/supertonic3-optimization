# 8차: 원본50 정상 속도에서 native2배속 품질 개선

사용자 요청: 정상 속도 원본50 기반으로 빠른 발화 정답을 만들어 먼저 검증한다. 추론 후처리 없이 모델 안에서2배속 누락을 해결한다. 실제8스텝과 기존 크기는 유지한다. 괜찮은 후보가 나오면 사용자가 청취한다. **개선 전 커밋 금지**, 개선 성공시 main에서 커밋한다. 기준 main은 `0e1702c`이며 이번 진행 내용은 아직 커밋하지 않았다.

## 교사와 정답 정책

원본은 고정 Supertone/supertonic-3 FP32 CFG4 최대50스텝만 사용한다. F1200은 학생 초기화일 뿐 정답이나 교사가 아니다. 원본 native2의 누락된 출력은 정답으로 쓰지 않는다. 최대품질 정상 출력에서 명시적인 학습용 속도 변환을 수행한다. 변환 음성은 원본 그대로의 출력이 아니며 그 차이를 기록한다. 학습 도구는 runtime에 연결하지 않는다.

먼저 기존 원본50 speed1.0의 KO/EN × M1/F1 4개 파형을 해시로 확인했다. WSOLA atempo2단 변환 후보는 KO1/50,EN0/22 오류였고, 여성 음성의 사용자 정의 유성음 harmonic/interharmonic 진단이 약4~5dB 낮아졌다. 더 정교한 참조를 비교하기 위해 R3 offline 방식을 추가했다. 해당 수치는 자연스러움 점수나 Paradee 수치의 재현이 아니다.

R3 후보4개는 개발 인식기에서 KO0/50·EN0/22, DTW 기준 유성음 F0 차이 중앙값 약20cents였다. 여전히 여성 음성의 harmonic 진단은 원본보다 약3.5dB 낮아졌으므로 지각적 동등성을 주장하지 않는다. 학습 이후 사용자의 청취가 필요하다. 근거: `results/round8_{fast,r3}_reference_pilot{,_asr,_acoustics}.json`.

R3 의존성은 GPL Rubber Band4.0.0, 원본 commit `1d95888bec3ae0a17c0c4af791810d5a63f6bc35`를 로컬 `.private/rubberband-v4`에 빌드했다. 바이너리는 배포하거나 inference 요구사항에 넣지 않는다. 공식 자료: https://github.com/breakfastquay/rubberband . Mac 빌드 설정은 release·shared·vDSP FFT·builtin resampler·optional features disabled다. 최신 Xcode의 size_t 헤더 문제로 C++ `-include stddef.h` 빌드 인자를 추가했다. Meson1.12.1·Ninja1.13.2를 프로젝트 가상환경에 설치했고 전역 시스템은 변경하지 않았다. 재현할 때 빌드와 Ninja 실행 양쪽에서 가상환경 bin을 PATH에 둔다.

## 학습 자료 생성과 필터

`distill/prepare_speed_audio.py`가 기존 원본50 학습 궤적의 최종 latent를 원본 FP32 decoder로 그대로 해독한다. 원본 duration predictor와 문장·음색으로 원래 speed1.05 길이를 재계산해 자른 뒤, **학습 정답에만** R3 압축을 수행한다. 목표 speed2.0이므로 압축 배율은2/1.05이며 정상 1.0의 절반 길이에 해당한다. native2의 예상 길이와 차이는0~1sample이다.

80개 train/16개 validation 총96개를 생성했고 각 원본 최종 latent 파일 해시, 정상/빠른 WAV 해시, 변환 설정을 `distill/fast-original50-round8/manifest.json`에 남겼다. 원본50 provenance와4개 원본 ONNX SHA를 확인한다. 문장·음색·seed 분할은 기존 궤적을 유지하며 train/validation을 섞지 않는다.

96개 빠른 정답 전부를 Whisper base INT8 CPU1로 검사했다. **오류0인66개만 사용**하며 train54/validation12다. 제외한30개는 학습에 넣지 않는다. 인식 오류0은 음색·잡음 품질 보장이 아니므로 별도 음향 진단과 청취를 유지한다. 이미 사용한 개발/검증 데이터이므로 새로운 독립 시험으로 주장하지 않는다.

초기 코드 동작 확인용 학습은 전체 정답 필터가 끝나기 전에 약15회까지 실행해 중단했다. 출력으로 채택/내보내기하지 않았고 후속 학습은 다시 원래 F1200에서 시작했다. 정식 trainer는 완전한96개 ASR audit과 문장 일치를 검사하고 오류0 정답만 허용한다. 원시 로그는 `results/round8_train_audio_v1.log`, 정식 필터 학습은 `round8_train_audio_filtered_v1.log`다.

## 모델 내부 학습

`distill/train_native_speed.py`는 F1200 초기화 후8회를 실제 rollout하고, 고정 원본 decoder를 통해 빠른 정답 파형에 MRSTFT·onset 손실을 적용한다. 빠른 음성 업데이트3회마다 정상 원본 업데이트1회를 섞으며, 정상 latent anchor를 추가한다. decoder·duration predictor·text encoder와 모델 구조는 고정한다. fast latent 단순 보간을 정답으로 사용하지 않는다.

현재 정식 실험은320회, lr1e-5,80회마다 validation을 측정한다. validation은 인식 오류0인 자료에서 KO/EN 각2개·서로 다른 음색을 선택한다. fast audio loss +0.3×normal audio loss로 체크포인트를 선택하지만, 손실 감소만으로 개선 성공을 선언하지 않는다. native8 생성 후 발음/정상 속도 회귀와 사용자의 청취를 확인해야 한다.

이 파일은 진행 기록이다. 최종 후보·성능·청취 결과는 실행 후 갱신한다. 현재 커밋/푸시·새 모델 업로드는 수행하지 않았다.

## 파형 학습과 읽기 위치 실험 결과

정식 audio-only 80/160/320회 모델을 FP32로 내보내 native8·speed2·M1/F1·KO/EN 개발 문장을 생성했다. 같은 로컬 비교의 기존 INT8 F1200은 KO22/50·EN10/22, audio80은22/50·6/22, audio160은15/50·4/22, audio320은26/50·5/22였다. 정상1.05의 같은 짧은 문장은 모두 KO0/50·EN0/22였다. FP32 후보를 배포할 후보로 확정하지 않았으며 INT8과의 정밀도 차이도 남아 있다. 160회의 부분 개선도 누락 해결이 아니고320회는 악화돼 커밋하지 않았다.

원본 정상50의 실제50회 궤적을 다시 실행해 마지막 text-attention4개를 수집했다. 저장된 원본 최종 latent와 비교해 선택된66개 모두 최대 오차0을 확인했다. 첫 수집에서 attention의 head/CFG batch 축을 잘못 골라 중단하고 폐기했다. 올바른 출력은 `[8heads,2CFGbranches,time,text]`이므로 조건 분기는 `[:, :1]`이다. 올바른 수집은 `distill/original50-attention-round8-v2`다. 이 실수의 자료는 학습에 쓰지 않았다.

원본50의 읽기 위치를 학생에게 생성 중 강제로 주입하는 oracle 대조군도 시험했다. 해당 방식은 매번 원본50을 필요로 하므로 배포 가능한 후보가 아니다. 정상 원본의 마지막 attention을8회 모두 강제하면 오히려 전혀 다른 문장으로 망가졌다. 원본/학생의 중간 표현을 그대로 교체할 수 있다는 가정은 성립하지 않았다(`round8_oracle_attention{,_asr}.json`).

직접 교체 대신 마지막4회에만 약한 KL0.1 위치 손실을 더하고, KO/EN을 균형 추출해 fast2회/normal1회로 학습했다. 원본 정답은 계속 동일하며 F1200 계열 체크포인트는 초기화로만 사용했다. attention160의 개발 결과는 KO13/50·EN4/22로 여전히 누락이 남았다. 정상 짧은 문장은0이었다. 전체240회 기록은 `distill/native-speed-round8-attention-v1/report.json`이며 이 실험도 개선 성공으로 선언하지 않았다.

## 빠른 latent 정답 복원과 직접 배속 조건

기존 그래프는 배속 입력 없이 duration만 나누어 길이를 줄인다. `distill/add_rate_condition.py`는 time-MLP의256차원 중간값에 `(speed-1.05)*learned_vector`를 더하는 입력을 추가한다. **추가 파라미터256개, denoising8회 그대로**, 크기 축소는 없다. 속도 입력이 추론 파형을 가공하지 않는다. `runtime.py`는 해당 입력을 요구하는 그래프에만 값을 전달하며 기존 모델은 무시한다.

첫 시제품은 MLP 마지막 출력64차원에256차원 벡터를 더하는 shape 오류가 있어 parity 검사에서 걸러졌다. 학습/배포하지 않았다. 수정본은 첫 MLP의256차원에 삽입한 `models/round8-rate-source-v2`이며, 초기 벡터0에서1.05/2.0 모두 기존 F1200 FP32와 bit-identical을 확인했다(`round8_rate_initial_parity.json`).

`distill/invert_fast_reference.py`는 원본50 최종 latent를6개 프레임 묶음에서 풀고 시간 보간한 값을 **초기값으로만** 사용한다. 이후 원본 decoder를 고정한 채 latent만300회 최적화해 검증된 R3 빠른 정답의 MRSTFT를 맞춘다. 단순 보간 자체를 정답으로 채택하지 않는다. 학생의 출력이나 학생 decoder는 정답 생성에 쓰지 않는다.

M1/F1·KO/EN 4개 pilot의 복원 오차가 약2.0→0.68~0.84로 줄었다. 복원 파형은4개 모두 개발 ASR 오류0, R3 정답 대비 F0 차이 중앙값0~10cents였다(`round8_inverse_pilot_{asr,acoustics}.json`). 이것은 학습 정답의 복원 검사이며8스텝 학생의 품질 개선 결과가 아니다.

이 검사를 통과한 정답을 확대해 train의 KO17개·EN17개, validation의 KO4개·EN4개 총42개의 latent를 복원하고 있다. 기존pilot4개는 해시 확인 후 재사용한다. 복원 파형도 다시 인식 검사를 통과해야 최종 flow 학습에 사용한다.

`distill/train_rate_flow.py`는 정상 원본50 궤적과, 복원한 빠른 정답 latent에 대한 합성 직선 flow 경로를 번갈아 학습한다. 빠른 경로의 noise는 업데이트마다 새로 뽑는다. 원본의 실제 확산 궤적이라고 표시하지 않는다. speed 조건256개는 lr5e-4, 기존 가중치는1e-5로 학습한다. 실제 생성은8회다. 먼저4개 pilot 문맥에서 학습 자체가 가능한지 확인하며, 이 overfit sanity check를 일반화 성능으로 해석하지 않는다. 이후 별도 validation 문장으로 확대해야 한다.

현재 main은 여전히 `0e1702c`다. 이번 변경·자료·중간 모델은 커밋/푸시/업로드하지 않았다.

## 8스텝 flow 및 디코더 내부 retiming 대조군

새 speed 조건을 붙인4문맥 pilot flow는250/500회에서 update MSE가 줄었지만 실제8회 생성의 발음이 나빴다. 합성 직선 궤적의 teacher-forced loss만으로 실제 rollout 품질이 보장되지 않았다. 이 모델은 채택하지 않았다. 복원 정답42개 중 원본 decoder 재합성 ASR 오류0인 것은 train32개(KO15/EN17), validation6개(KO2/EN4)였다. 나머지4개도 후속 학습에서 제외한다.

별도 작은414,488파라미터의 pre-decoder latent head를 시험했다. 입력은 고정 F1200의 정상8회 latent이며 **학생은 입력 생성에만 사용**, 정답은 계속 원본50→R3→고정 원본decoder inversion이다. head는6프레임 묶음을24채널의 세부 시간축으로 풀어 압축하고 신경망 잔차를 더한다. 정상 길이가 같으면 잔차 gate0이어서 원래 decoder 입력을 그대로 보존한다.

학습32/validation6 문맥,3000회 head의 짧은 native2 검사 결과는 KO10/50·EN1/22였다. 정상1.05의4개 파형은 기존 F1200과 bit-identical이었다. 추가 waveform-only3000회 모델도 KO5/50·EN2/22로 아직 완전한 복구가 아니었다. 원본 대비 F0 차이 일부는 정상1.0/1.05의 기존 prosody 차이도 섞이므로 head만의 pitch 변형으로 단정하지 않는다.

**학습 잔차를0으로 둔 단순 내부 시간축 압축 대조군이 가장 좋은 결과였다.** M1/F1·KO/EN4개 모두2배속 ASR 오류0. 이때 실제 입력인 F1200 speed1.05와 비교한 F0 차이 중앙값은10~20cents였다. 사용자 정의 harmonic/interharmonic 진단은 일부 약3dB 낮아져 잡음 청취를 확인해야 한다. 아직 사용자의 청취나 서버 성능 검사를 통과한 것으로 선언하지 않는다.

이 경로는 정상 길이8회에서 완전한 phonetic latent를 만든 뒤, **보코더 ONNX 내부**에서 묶음6개를 풀어24채널 세부 프레임을 압축하고 파형을 직접 만든다. 오디오를 만든 뒤 tempo/pitch 필터를 적용하지 않는다. 가중치 추가·축소·denoising 생략은 없다. 대신 더 긴 latent를8회 계산하므로 native2보다 계산량이 증가한다. 실제 RTF와 품질의 절충을 반드시 보고한다.

`prepare_latent_retime.py`는 학습 잔차/CNN을 제거해 추가 파라미터0인 이 변환만 원본 decoder에 결합한다. 기존 F1200 vector는 그대로다. 초기 ONNX Resize의 PyTorch/ORT 비교는 단위 진폭 random 입력에서 좌표 FP32 반올림 차이가 최대약1.6e-4로 엄격한 parity 한계에 걸렸다. 정상 길이 동일 경로는 정확히 같았다. 단순히 허용치를 넓히는 대신 half-pixel 좌표·Gather·선형 가중합을 명시해 **5개 동적 길이 모두 Torch/ORT 최대 오차0**을 확인했다. 준비 도중 오류가 난 모델/평가는 폐기·표시했고, 생성은 staging 디렉터리에서 성공 후에만 최종 폴더로 옮기도록 수정했다. 유효한 모델은 `models/round8-latent-retime`이다.

8회 중간에 길이를 바꾸는 다른 대조군도 시험했다. 첫 k회 정상 길이, 나머지8-k회 짧은 길이를 쓰며, 알려진 초기 noise를 빼고 새 길이 noise를 넣는 보정/단순 보간을 비교했다. k0 보정 대조군4개는 기존 native2 파형과 bit-identical로 프로토콜을 확인했다. k2/4/6은 누락·발음 문제가 남았고, k8(모든8회 뒤 decoder 내부 retiming)이 가장 좋았다. 이 시간표 변경은 현재 기본 runtime에 넣지 않았다.

현재 추가 검사: 내부 생성 속도1.05/1.2/1.35/1.5와 출력2.0의 품질·실제8회 카운트, 장문, 정상파형 보존, 그리고 기존1코어/1GB 서버 RTF를 확인한다. 새 모델·자료 업로드와 커밋/푸시는 계속 보류한다.

## 청취 확인과 선택 설정: 내부1.35 → 출력2.0

사용자는 처음 내부1.05 후보에서 **누락·뭉개짐은 없지만 남성 기계음이 강해졌다**고 피드백했다. 이를 기분 탓으로 처리하지 않고 품질 결함으로 기록했다. 내부 생성 속도1.05/1.2/1.35/1.5를 비교했다. 출력은 항상2.0이고 실제 denoising 호출은 매번8회였다. 짧은 M1/F1·KO/EN4개에서1.05/1.2/1.35는 모두 KO0/50·EN0/22,1.5는 KO2/50·EN2/22였다.

내부1.35는 압축 비율이2/1.35≈1.48로1.05 설정의2/1.05≈1.90보다 작다. 남성 한국어의 사용자 정의 harmonic/interharmonic 진단값은0.255→2.419dB였다. 이 수치는 MOS나 Paradee 지표 재현이 아니며 개선의 단독 근거로 쓰지 않았다. 실제 비교 음성을 제시했고 사용자는 **“기계음이 줄었고 발음도 괜찮음”**으로 확인했다. 음색·문장 한정 청취 결과다(`round8_retime_human_feedback.json`).

이 확인 후 내부1.35/출력2.0을 **별도 선택 실행하는 개선 후보**로 기록한다. F1200 기본 경로와 학습된 가중치는 그대로다. 새 학습 모델로 표시하지 않는다. `prepare_latent_retime.py` 기본 내부 속도는1.35이며 공개된 pinned F1200만으로 재학습 없이 만든다. 추론은 추가 파라미터0·실제8회이며 waveform 후처리가 없다.2.0 이외 배속은 기존 duration 경로다.1.35 설정을 모든 배속/음색의 최적값으로 주장하지 않는다.

학습 head의 잔차를0.1로 줄인 대조군도 시험했다. 짧은4개 ASR은0이었지만 남성 KO 진단값은0.120dB로0 잔차의0.255보다 개선되지 않았다. 이를 선택하지 않았고 새 청취 후보를 불필요하게 늘리지 않았다. 학습 head 전체/부분 잔차 checkpoint는 배포 대상으로 정하지 않았다.

## 확대 검증: 해결된 범위와 남은 오류

Whisper base INT8·CPU1·beam5, 기존에 소비한16개 validation 문맥(언어별8개, 여러 음색), seed42. 한국어는 정규화 문자 CER, 영어는 단어 WER. 새 독립 시험이 아니며 ASR은 음질·화자 유사도 점수가 아니다.

| 경로 | KO 오류/문자 | EN 오류/단어 |
|---|---:|---:|
| F1200 정상1.05 |5/193 (2.59%)|0/96|
| 기존 F1200 native2 |133/193 (68.91%)|68/96 (70.83%)|
| 내부1.05 → 출력2 |42/193 (21.76%)|2/96 (2.08%)|
| 선택 내부1.35 → 출력2 |51/193 (26.42%)|1/96 (1.04%)|

선택 후보는 기존 native2보다 크게 개선됐지만 정상 속도 수준에는 못 미친다.1.35가1.05보다 모든 지표에서 좋은 것도 아니다. 남성 청취·짧은 발음·계산량과 확대 한국어 오류 사이의 절충이다. **전 문장 누락 해결·원본 음질 일치·완전한 배속 지원이라고 선언하지 않는다.** 후속 최우선 과제는 validation 한국어의 자음·음절 보존이다.

소비한 장문 M1/F1·KO/EN4개: 내부1.05는KO18/180·EN0/86, 내부1.35는KO13/180·EN1/86이다. 짧은 표본만으로 장문을 대신하지 않았다. 원시 출력은 `round8_retime_g135_{validation,long}_asr.json`과 각 대조군에 보존한다.

이 과정에서 train/validation의 동일 case 번호가 음성 파일을 덮어쓰는 경로 문제가 발견됐다. 해당 전체 corpus 실행은 `.private/round8_g135_all_corpus_invalid_path_collision.json`으로 격리해 결과로 사용하지 않았다. 평가기는 split 접두사를 추가했고 위16개 validation은 고유 경로로 재생성했다. 준비 실패 중 부분 생성물로 실행한 이전 평가도 폐기했다. 실패/불완전 실행은 성능 주장에 넣지 않는다.

## 실제 EPYC1코어·1GB 측정

기존 허용 서버, EPYC7551, 논리코어0 고정, ORT/BLAS1스레드, nice19, 실제8회, 공유 서버. F1200+native 엔진v2+조건 캐시, 모델 로딩·파일 저장 제외, 문장별 warm-up1회. 짧은 문장은2회 중앙값, 긴 문장은1회라 통계적 안정성을 주장하지 않는다.50ms process-tree RSS 감시480MB·시스템 여유80MB, 다른 프로세스는 건드리지 않았다.

| 내부 생성 → 출력 | 문장 | 생성초 | RTF |
|---|---|---:|---:|
|1.05 →2|KO 짧음|4.707|1.685|
|1.05 →2|EN 짧음|4.351|1.680|
|1.35 →2|KO 짧음|3.950|1.414|
|1.35 →2|EN 짧음|3.610|1.394|
|1.35 →2|KO 김|13.016|1.407|
|1.35 →2|EN 김|12.917|1.455|

내부1.35는1.05보다 짧은 생성 시간이 약16.1%/17.0% 줄었다.2배속 결과 길이 기준이므로 RTF는1보다 크며 **실시간 미달**이다. 원래 native2보다 긴 latent를8회 계산한다는 비용을 숨기지 않는다. 짧은 후보 표본 최대RSS367.12MB, 장문397.68MB, 감시 중단 없음. 원시 서버 JSON과 guard는 `round8_retime_g105_server*`, `round8_retime_g135_server*`, `round8_retime_g135_long_server*`다.1.05 대조군 peak370.78MB다. ARM 성능으로 환산하지 않는다.

원격에는 별도 후보 폴더를 만들고 원래 엔진v2 패키지의 가중치를 유지했다. 기존 공유 파일은 원자적 파일 교체로 분리했다. 서버 개인 자료·키·주소는 공개 기록에 넣지 않는다.

## 회귀 검사와 이어서 할 일

기존 원본 공식 실행기와 비교하는 `tests_runtime.py`2개 검사가 통과했다. 준비한 retime 그래프의5가지 동적 길이 Torch/ORT 차이는0이며 정상 길이 identity가 유지된다. `experiments/check_latent_retime.py`는 M1/F1·KO/EN4개에서 정상1.05 float32 파형이 기존 F1200과 완전히 같고,2.0은 실제8회 호출하며 길이 비율1.05/2를 지킴을 확인했다(`round8_retime_g135_regression.json`).

최소 사용·재현 명령은 README와 `docs/REPRODUCING.md`에 있다. 선택 후보는 새 체크포인트/수집 데이터 없이 기존 공개 F1200에서 준비되므로 새 HF 업로드가 필요 없다. 실패한 학습 실험 스크립트와 수치도 함께 보존하되 중간 가중치·WAV·NPZ·개인 자료는 Git에 넣지 않는다.

학습 실험을 다시 할 때는 `requirements-speed-train.txt`와 pinned 원본50 자료가 필요하다. `distill/prepare_speed_audio.py --source <원본50 궤적>` → `experiments/asr_manifest.py` → `distill/invert_fast_reference.py` → 재인식 필터 순서를 지킨다. 원본50 이후의 R3 변환은 학습 정답에만 쓰며 원본 그대로인 것처럼 표시하지 않는다. R3 소스는 https://github.com/breakfastquay/rubberband 의 v4.0.0, commit `1d95888bec3ae0a17c0c4af791810d5a63f6bc35`다. 로컬 release shared build의 vDSP FFT·builtin resampler를 사용했고, 최신 Xcode의 헤더 문제에 `-include stddef.h`를 지정했다. 라이브러리는 배포/추론에 포함하지 않으며 Linux에서는 해당 OS용 빌드 경로를 `rubberband_reference`에 전달해야 한다. 실패한 학습을 재현하기 위한 별도 의존성이고 선택 retime 후보에는 불필요하다.

후속 순서: (1) validation 한국어의 잔여 오류를 청취·정렬해 자음 손실과 인식기 오류 구분, (2) 현재 품질을 보존하는 보코더 내부 표현 개선, (3) 같은 서버 조건의 반복 측정과 정상 출력 보존, (4) 새 독립 문장은 후보 선택을 끝낸 후 사용. 기존 학생을 새 교사로 삼거나8회 미만으로 줄이지 않는다.

실제 서버에서 저장한 선택 후보 M1 WAV도 로컬에 가져와 같은 인식기로 검사했다. 짧은KO0/25·EN0/11, 긴KO5/90·EN0/43이었다(`round8_retime_g135_server_asr.json`). 로컬 M1 장문 결과와 비트/ASR 완전 일치를 주장하지 않는다. float 커널·환경 차이가 있으며 서버의 실제 출력 검사를 별도로 보존한다. 장문 한국어 오류는 해결 과제로 남는다.
