# 최종 보관 상태와 복원 방법

2026-11-02 사용자의 요청으로 개발·성능 실험을 종료하고 생성한 로컬/검증 서버의 대용량 산출물을 정리했다. 기준 가중치는 **F1200**, 추론은 **실제 denoising 8회·한 스레드**이다. 정리 과정에서 학습·추가 최적화·음질 변경은 하지 않았다.

## 공개 보관본

모델 저장소의 최종 고정 revision은 **`cbef65027ca699895cdeb1acd3308eeeee2a7a63`**이다. 이전 모델·재현 자료 139개 파일은 변경하지 않고 최종 배포본을 추가했다.

| 자료 | 위치 | 사용 조건 |
| --- | --- | --- |
| 최종 Linux v5 내보낸 모델 | [F1200-retime-v5](https://huggingface.co/babihoba/supertonic3-optimization/tree/cbef65027ca699895cdeb1acd3308eeeee2a7a63/models/F1200-retime-v5) | Linux x86_64 AVX2. native 정수 커널은 대상 기기에서 빌드 |
| 최종 Mac ARM64 내보낸 모델 | [F1200-retime-arm64](https://huggingface.co/babihoba/supertonic3-optimization/tree/cbef65027ca699895cdeb1acd3308eeeee2a7a63/models/F1200-retime-arm64) | Mac M4 Max에서 검증한 전체 정규화 패키지. native 커널 빌드 필요. Oracle ARM 성능은 미검증 |
| 표준 ORT F1200 | [round5-original50-student](https://huggingface.co/babihoba/supertonic3-optimization/tree/cbef65027ca699895cdeb1acd3308eeeee2a7a63/models/round5-original50-student) | 별도 native 커널 없이 실행 |
| 표준 ORT 내부 2배속 패키지 | [round8-latent-retime-g135](https://huggingface.co/babihoba/supertonic3-optimization/tree/cbef65027ca699895cdeb1acd3308eeeee2a7a63/models/round8-latent-retime-g135) | 내부 1.35× → 출력 2×의 기존 경로 |
| 원본50 학습 초기·최종 체크포인트 | [reproduction/round5](https://huggingface.co/babihoba/supertonic3-optimization/tree/cbef65027ca699895cdeb1acd3308eeeee2a7a63/reproduction/round5) | F1200 재현용 |
| 교사 데이터·manifest | [dataset round5](https://huggingface.co/datasets/babihoba/supertonic3-optimization-data/tree/e56d93c9959647c850b639a614e629e4e4ac1593/round5) | 원본 FP32·CFG4·50스텝 기반의 기존 학습/검증 데이터 |

새로운 학습 가중치나 학습 데이터는 만들지 않았다. 최종 두 내보낸 패키지는 **44파일·387,707,191 bytes**이며 음색·설정·라이선스·전체 ONNX가 포함된다. CPU별 compiled binary는 포함하지 않는다. 모델별 SHA-256은 [최종 파일 목록](models/final_exports_manifest.json)에 있다.

정리 전에 모델·필수 체크포인트·교사 데이터의 **고유 30개 공개 객체, 865,247,478 bytes**를 인증 없이 실제 다운로드하여 해시를 확인했다. 필수 초기/최종 체크포인트 2개와 풀린 교사 데이터/metadata 770개도 로컬과 공개 manifest를 대조했다. [검증 영수증](../results/closeout_2026-11-02_hf_verification.json)

## 필요한 환경만 다시 만들기

Python 3.12 이상과 각 기기의 native 빌드 도구를 사용한다. 다운로드에는 Hugging Face 토큰이 필요하지 않다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Linux x86_64 AVX2에서 최종 v5 패키지를 복원한다.

```bash
python download_artifacts.py --model F1200-retime-v5
python -m pip install ziglang==0.17.0
python native/build_exact.py --cpu znver1
python synthesize.py '안녕하세요. 보관한 모델을 실행합니다.' \
  --model .hf-assets/model/models/F1200-retime-v5 \
  --lang ko --voice M1 --steps 8 --speed 1.0 --output samples/restored.wav
```

Mac ARM64에서 최종 정규화 패키지를 복원한다.

```bash
python download_artifacts.py --model F1200-retime-arm64
python native/build_row_norm.py
python synthesize.py '안녕하세요. 보관한 모델을 실행합니다.' \
  --model .hf-assets/model/models/F1200-retime-arm64 \
  --lang ko --voice M1 --steps 8 --speed 1.0 --output samples/restored.wav
```

별도 native 커널 없이 기본 F1200을 쓰려면 `python download_artifacts.py --purpose inference`로 받고 `.hf-assets/model/models/round5-original50-student`를 지정한다. 초기/최종 체크포인트와 교사 데이터는 `python download_artifacts.py --purpose reproduce --model F1200-retime-v5`로 복원한다. 추가 학습의 전체 절차는 [재현 안내](REPRODUCING.md)에 남아 있다.

## 정리한 파일과 유지한 기록

- 로컬: 다운로드/파생 모델, 학습 산출물과 수집 데이터, 생성 음성, 배포 archive 중복본, 임시 검증 폴더, native 빌드 결과, 가상환경, 캐시를 정리했다. 기존 Git 추적 파일 937개와 개인 접속키는 보존했다.
- 검증 서버: 기존 작업 전용 디렉터리 안의 모델·샘플·실험 가상환경·wheel·실험 바이너리만 정리했다. 작은 소스·결과와 메모 archive는 남겼다. 다른 디렉터리와 개인 프로그램은 변경하지 않았다.
- Git에는 소스·평가·실험 이력·정리 영수증을 보존한다. 비공개 메모는 로컬 `local-archive/`에 약 93MB의 압축본으로 남겼고 공개 업로드하지 않았다. 이 폴더와 접속키는 Git에서 제외한다.
- 로컬 청취 포트 7788/57727 및 이 작업의 추론·실험 프로세스가 실행 중이지 않음을 확인했다. 추가 작업은 예약하지 않았다.

로컬에서 관측한 여유 공간 증가는 **12,008,103,936 bytes**, 서버에서는 **7,707,914,240 bytes**이다. 논리 파일 용량은 hardlink·APFS 등의 영향으로 실제 회수 공간과 다르므로 구분해 기록했다. [로컬 정리](../results/closeout_2026-11-02_local_cleanup.json) · [서버 정리](../results/closeout_2026-11-02_server_cleanup.json)

## 성능 기록의 해석

마지막 버전별 RTF 비교는 [18차 기록](../records/2026-10-30/ROUND18_REMEASUREMENT.md)이다. 해당 서버는 다른 프로세스가 함께 실행되는 1코어·약1GB의 저사양 공유 환경으로, 반복 값과 비교 순위의 일관성이 낮다. 작은 차이의 확정적인 우열이나 지속 실시간 보장으로 해석하지 않는다. 이 수치를 Oracle ARM으로 환산하지 않는다. 정리하면서 새로운 성능·WER/CER/MOS 실험은 실행하지 않았다.
