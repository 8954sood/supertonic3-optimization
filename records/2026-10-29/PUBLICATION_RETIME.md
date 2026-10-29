# 선택2배속 패키지 공개

사용자가 필요한 새 모델도 Hugging Face에 올리도록 요청했다. 엔진v4 자체는 학습 가중치가 그대로여서 모델 갱신을 필요로 하지 않는다. 다만8차에서 청취한2배속 ONNX 보코더 그래프는 아직 공개되지 않았으므로 완성된 portable 패키지를 별도 경로에 추가했다. 기존 F1200과 데이터는 보존한다.

- 모델 저장소: https://huggingface.co/babihoba/supertonic3-optimization
- 새 고정 revision: `f1d05bdb691d8769557e19644f3a63089992daf0`
- 패키지: `models/round8-latent-retime-g135/`,21파일,193,741,035bytes
- 이전 revision: `d33f1094c6057d8c613f2c45987ba45e57c4b3af`
- 변환·선택 엔진 source commit: `7704219`
- 교사 데이터: 기존 `babihoba/supertonic3-optimization-data@e56d93c9959647c850b639a614e629e4e4ac1593`, `round5/`

F1200의 duration predictor·text encoder·vector estimator3개는 기존 공개 파일을 서버 내 복사했다. 새 vocoder 그래프와 manifest·작은 설정 파일/설명만 업로드했다. 총21개 패키지 파일을 size+LFS SHA256 또는 Git blob hash로 확인했다. 기존118개 파일(README/asset manifest 제외)의 blob ID가 모두 그대로임을 확인했다. 새 vocoder와 manifest는 인증 없이 새 고정 revision에서 실제 내려받아 SHA256을 확인했다. 검증 다운로드의 중복 vocoder는 삭제해 로컬 디스크 사용 증가를 제한했다.

가중치 학습·새 checkpoint·새 정답 데이터는 없다. 새 보코더는24채널 세부 latent 시간축을 내부에서 압축하고 기존 학습 decoder로 파형을 만든다. inference는 내부1.35/출력2.0·실제8회이며 후처리가 없다. 다른 배속은 기존 duration 경로다. 공개된 원본50 초기/최종 checkpoint와 데이터가 계속 F1200 재현의 원천이다. 실패한8차 학습 모델이나 중간 checkpoint는 업로드하지 않았다.

모델 카드·manifest에 짧은 남성 청취의 한정된 개선, 확대KO51/193·EN1/96 오류, 정상 속도와의 차이, EPYC1코어/1GB warm RTF, ARM 미측정과 실시간 미달을 명시했다. 범용2배속 누락 해결이나 새 품질 점수로 과장하지 않는다. 모델 라이선스와10개 음색을 보존했고 optional AVX2 엔진은 binary를 포함하지 않고 GitHub source에서 별도 빌드한다.

`download_artifacts.py`는 새 모델명을 지원하고 새 revision에 고정한다. 기본 선택은 F1200 그대로다. 새2배속 후보의 재현 자료 선택은 원본50 계열로 분류해 이전 B 데이터셋을 잘못 내려받지 않도록 했다. 패키지/교사 데이터 링크와 명령은 README와재현 안내에 연결했다.

원시 공개 검증: `results/round9_hf_upload_verification.json`. 사용자 요청에 따라 main commit 이후의 추가 RTF 실험도 계속하며, 배속 설정이나 학습 가중치를 변경해 속도를 얻지 않는다.

GitHub의 새 다운로드 명령도 토큰 없이 실제 실행해 패키지21개 파일 전체의 size/SHA256을 검사했다. 내려받은 패키지로 KO/EN2개를8스텝·2배속 생성했고 로컬 선택 후보와 float32 파형이 완전히 같았다. 확인 후 이 중복 다운로드 폴더는 삭제했다(`results/round9_hf_public_download.json`).
