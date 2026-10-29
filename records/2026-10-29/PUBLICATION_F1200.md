# F1200 공개 모델·재현 자료 업로드

사용자가 2026-10-29에 성공한 최적화의 커밋과 기존 모델 업로드를 다시 허용하고 F1200을 기준 모델로 선택했다. 이전 업로드 중지 이력은 삭제하지 않으며 이번 허용 이후의 작업과 구분한다. 모델 구조·가중치는 추가로 축소/학습하지 않는다.

- 모델 고정 revision: `d33f1094c6057d8c613f2c45987ba45e57c4b3af`, https://huggingface.co/babihoba/supertonic3-optimization
- 데이터 고정 revision: `e56d93c9959647c850b639a614e629e4e4ac1593`, https://huggingface.co/datasets/babihoba/supertonic3-optimization-data
- 학습/출처 코드 커밋: `13fb103`. 원격 Git 푸시 여부는 별도 상태에서 확인한다.

추가한 모델 자산은 F1200 완전한 ONNX 패키지, E400 청취 비교 패키지, 원본50 학습 초기/최종 체크포인트 두 개와 보고서다. E400 훈련 checkpoint와 매200회 중간 checkpoint, 폐기한 B교사 재증류 결과, 사용자에게 거부된 C/D 모델은 추가하지 않았다. 기존 공개 B 및 과거 모델/데이터는 삭제하지 않았다. 추가 경로의 논리 용량은 모델726,286,972bytes 및 데이터143,492,132bytes이며 공통 모델 파일은 기존과 같은 해시다. Hub의 실제 물리 저장량은 별도로 측정하지 않았다.

데이터 archive는 original FP32 CFG4 50스텝 교사에서 생성한640train/128validation 쌍과 manifest다. 고유 문장은80/16개다. tar member는 상대 경로·일반 파일만 허용하고 소유자/기기 경로를 넣지 않았다. 공개 데이터는 synthetic latent이며 실제 녹음이 아니다. 모델/데이터의 upstream OpenRAIL-M 조건을 보존했다.

## 검증 완료

1. 새 모델 commit의50개 추가/갱신 파일과 데이터 commit의5개 파일을 Hub 메타데이터의 크기 및 LFS SHA256/Git blob hash와 대조했다. 모두 일치했다.
2. 인증 없이 고정 revision에서 F1200·재현 자료30개를 다시 내려받았다. 추가로 풀린769개 파일의 SHA256도 모두 확인했다.
3. 공개 F1200과 로컬 동결 F1200의 한국어 float32 파형이 샘플 단위로 완전히 같았다. 이는 Mac에서 실행한 숫자 동일성 확인이며 서버 속도/독립 품질 검사가 아니다.
4. 공개 최종1200 checkpoint를 다시 export해 duration/text/vector/vocoder 네 ONNX SHA256이 공개 배포본과 모두 일치했다.

원시 기록은 `results/round6_hf_upload_verification.json`, `results/round6_public_reproduction_verification.json`이다. 다운로드 도구의 기본 선택을 F1200으로 바꾸고 새 고정 revision을 지정했다. `--purpose reproduce`는 필요한 초기/최종 checkpoint와 original50 데이터만 받으며, archive를 안전하게 풀어 개별 텐서 해시도 확인한다. 모델마다 데이터 경로를 분리하여 과거B 자료와 섞지 않는다.

Git에는 소스·작은 JSON·문서만 넣는다. 가중치/음성/archive/인증정보는 제외한다. write token은 숨김 입력으로 프로세스 메모리에만 전달했고 파일/커밋에 저장하지 않았다. 새 독립 ASR 평가나 청취 동등성 백분율을 주장하지 않는다. 기존 개발 pilot은 KO0/44, EN0/20이고 모든 모델에서 같았으나 표본이 매우 작다.
