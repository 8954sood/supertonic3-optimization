# Hugging Face 업로드 완료와 재현 확인 — 2026-10-29

모델: https://huggingface.co/babihoba/supertonic3-optimization

데이터: https://huggingface.co/datasets/babihoba/supertonic3-optimization-data

모델·중간 checkpoint 67개와 교사 궤적/manifest 1,665개를 업로드했다. Hub의 파일 크기와 LFS SHA-256 또는 Git blob hash를 로컬 목록과 대조했으며 누락·불일치가 없었다. 모델 고정 revision은 `85bc100ca8d7cefad17b63a4c7ba489638dbe2f2`, 데이터 완료 revision은 `817056ae3e9b8fa1a2e464a212bb2ba1f4c5196a`다. README/라이선스가 이후 갱신되더라도 이 검증은 위 revision의 자산을 대상으로 한다.

기존 업로더의 Xet 전송이 작은 NPZ 묶음에서 정체됐다. 분리된 Hugging Face Hub 1.5.0 환경에서 `HF_HUB_DISABLE_XET=1`로 남은 416개를 HTTP LFS 업로드해 완료했다. 추론·학습 환경은 변경하지 않았다. 인증 token은 숨김 입력으로 전달하고 파일이나 Git에 저장하지 않았다.

`download_artifacts.py`는 고정 revision에서 선택 모델만 약188MB, 학습 재현 필수 checkpoint와 궤적만 약646MB 내려받고 크기·SHA-256을 검증한다. 전체 세 모델과 비교 자료를 받는 불필요한 다운로드를 줄였다. 공개 저장소 다운로드에는 token을 사용하지 않는다.

## 실제 확인

- 공개 모델 21개 파일을 직접 다운로드·해시 검증하고 한국어 한 문장을 합성했다. 로컬 보존 모델과 파형이 완전히 같았다.
- 공개한 600회 FP32 checkpoint를 다시 ONNX로 내보냈다. duration predictor·text encoder·vector estimator·vocoder 네 파일의 SHA-256이 배포본과 전부 일치했다.
- 공개 초기 checkpoint와 train1개/validation3개 궤적을 직접 다운로드·해시 검증했다. 원래 작업 경로가 없는 위치에서 학습 1회를 수행했고 검증 update 상대MSE는0.1099070385→0.1073635146이었다. 전체 훈련의 장비 간 바이트 단위 재현을 보장하는 검사는 아니다.
- 원본 교사 graph의 미분 가능 구현은 공개된 train/validation 텐서의12개 vector/decoder 조건에서 기존1e-4 상대RMSE 제한을 통과했다.

`distill/train_joint.py`가 manifest의 이전 `file` 경로를 그대로 읽던 문제를 수정했다. 현재는 지정한 `--data` 아래의 split/파일명을 사용하므로 `.hf-assets/data/trajectories-v2`로 다운로드해도 동작한다. `distill/validate_graphs.py`에도 데이터·출력 경로 옵션을 추가했다.

원시 확인: `results/round4_hf_upload_verification.json`, `results/round4_public_download_verification.json`, `results/round4_reexport_verification.json`, `results/round4_public_reproduction_verification.json`, `results/round4_relocation_verification.json`, `results/round4_published_graph_parity.json`.

ARM 실측은 사용자 제공 [PR #1](https://github.com/8954sood/supertonic3-optimization/pull/1)을 검토했다. 이 작업에서 ARM에 새로 접속하거나 재측정하지 않았다. 후속 후보는 사용자의 요청에 따라 기존 EPYC 1코어·1GB 서버에서 평가한다.
