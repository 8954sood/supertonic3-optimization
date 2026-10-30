# HTTP 서버와 스트리밍

원본 Supertonic Python 1.3.1의 서버 요청 형식에 맞춰, 추론 부분을 이 저장소의 `runtime.Engine`으로 제공합니다. 모델은 시작할 때 한 번만 로드하며 음색을 바꿔도 ONNX 세션을 새로 만들지 않습니다. **추론 한 스레드·완료된 텍스트 조각마다 실제 8스텝**을 유지합니다.

## 실행

```bash
python -m pip install -r requirements-serve.txt
python download_artifacts.py --purpose inference --model round8-latent-retime-g135
python serve.py --model .hf-assets/model/models/round8-latent-retime-g135
```

- 청취 화면: <http://127.0.0.1:7788/>
- API 문서: <http://127.0.0.1:7788/docs>
- 상태 확인: <http://127.0.0.1:7788/v1/health>

정배속만 필요하면 기본 F1200 패키지를 다운로드하고 `python serve.py`로 실행해도 됩니다. 이미 준비한 패키지는 `--model models/round8-latent-retime-g135`처럼 지정합니다. 자동 다운로드나 모델 변환은 서버 시작 시 수행하지 않습니다. Linux AVX2용 exact 패키지도 같은 인자로 지정할 수 있으며, Mac·ARM은 이식 가능한 표준 ORT 패키지를 사용합니다.

### 표준 ORT 조건 캐시 선택

같은 문장 안에서 변하지 않는 조건을 한 번 계산해 8스텝이 재사용하는 패키지를 준비할 수 있습니다. 서버 코드를 바꾸거나 가중치를 재학습하지 않습니다. 위에서 다운로드한 패키지에 한 번만 적용하세요.

```bash
python prepare_condition_cache.py \
  --source .hf-assets/model/models/round8-latent-retime-g135 \
  --output models/F1200-portable-cached
python serve.py --model models/F1200-portable-cached
```

기본 F1200에도 같은 도구를 사용할 수 있습니다. `--source`를 다운로드한 `round5-original50-student`로 바꾸세요. 결과 패키지는 원본과 별도 폴더로 만들고, 2배속 보정 등 원본 설정을 유지합니다. 기본은 파일을 복사합니다. 디스크를 아끼려면 준비 명령에 `--link-assets`를 붙여 변경하지 않은 파일을 하드 링크로 공유할 수 있습니다. 이때 두 패키지의 파일을 직접 덮어쓰지 마세요. 서로 다른 파일시스템에서는 복사로 전환합니다.

준비 과정은 ONNX 그래프·가중치 전체를 읽으므로 RAM이 넉넉한 개발 환경에서 수행하세요. Linux v4 등 `condition.onnx`가 이미 있는 패키지는 중복 변환을 거부합니다. 공개된 같은 가중치로 재현하는 엔진 변경이므로 추가 Hugging Face 모델 다운로드는 필요하지 않습니다.

M4 Max·추론 한 스레드에서 원래 retime 패키지와 교대로 비교한 실제 TCP 스트리밍 8조합의 전체 응답 시간 감소율 중앙값은 **3.5%**였습니다. 남성 한국어 1.0×는 차이가 없었고, 다른 7조합은 약 1.1~6.0% 줄었습니다. 같은 입력의 PCM은 전부 일치했습니다. ARM에서의 개선율은 아직 측정하지 않았습니다. [12차 기록](../records/2026-10-30/ROUND12_ENGINE.md) · [원시 HTTP 비교](../results/round12_paired_http.json)

### ARM64 행 병렬 정규화 선택

Mac ARM64에서 더 빠르게 실행하려면 위에서 준비한 조건 캐시 패키지에 정규화 커널을 적용할 수 있습니다. C 컴파일러가 필요하며, 바이너리는 사용할 기기에서 빌드합니다.

```bash
python native/build_row_norm.py
python prepare_row_norm.py \
  --source models/F1200-portable-cached \
  --output models/F1200-arm64 --all-parts --link-assets
python serve.py --model models/F1200-arm64
```

`--all-parts`는 벡터 추정기 36곳, duration 처리기 10곳, 텍스트 처리기 15곳, 보코더 10곳의 총 71개 정규화를 바꿉니다. 독립된 16개 행을 함께 계산하고, 남은 구간은 8행·4행·개별 행으로 처리합니다. 변환 전에 크기 정보를 추론해 저장하므로 주요 행렬곱·GELU의 기존 융합 경로를 보존합니다. 한 행 안의 연산 순서, 가중치, 실제 denoising 8회와 배속 경로는 유지합니다. `--link-assets`를 쓰면 변경하지 않은 파일을 공유하므로 두 패키지를 읽기 전용으로 취급하세요. 이를 빼면 파일을 복사합니다. 기존 벡터 추정기 전용 패키지에도 아래처럼 별도 폴더로 추가 변환할 수 있습니다. 소스를 받은 뒤 커널을 재빌드하고 서버를 재시작하세요. `--rows 4/8/16/32`로 비교용 빌드도 가능하며 기본값은 검증한 16입니다.

```bash
python prepare_row_norm.py --source models/F1200-arm64 \
  --output models/F1200-arm64-allparts --all-parts --link-assets
python serve.py --model models/F1200-arm64-allparts
```

이미 `--all-parts`로 만든 패키지는 중복 변환할 수 없습니다. 기존 벡터 추정기만 바꾸려면 이 옵션을 생략하면 됩니다.

시작 시 설치된 ORT와 네이티브 정규화의 결과를 폭 64·256·512에 대해 샘플 단위로 비교합니다. 호환되지 않거나 컴파일하지 않은 경우 명확한 오류로 중단하므로, 그 기기에서는 앞의 표준 ORT 패키지를 선택하세요. 기본 표준 패키지는 이 라이브러리를 요구하지 않습니다. Linux ARM64용 빌드 경로도 제공하지만 실기기 검증은 아직 없으며, Linux x86 AVX2 v4에는 이 커널을 적용하지 않습니다.

M4 Max·CPU 한 스레드·8스텝에서 직전 조건 캐시 패키지와 비교했을 때 실제 HTTP 전체 응답 시간 감소율 중앙값은 **9.04%**였고 8조합 모두 개선됐습니다. 생성 시간의 별도 12조합 비교는 **8.24%** 감소했습니다. 파형·PCM 일치, 취소, 1.25/1.30 등 배속 검사 결과는 [13차 기록](../records/2026-10-30/ROUND13_ENGINE.md)에 있습니다. 가중치는 그대로라 HF 모델을 새로 다운로드할 필요가 없습니다.

15차의 전체 정규화는 직전 16행 벡터 전용 패키지보다 생성 시간 **4.73%**, 실제 HTTP 전체 시간 **4.47%**, 첫 오디오 도착 시간 **4.77%**를 추가로 줄였습니다. 공개 retime 표준 패키지와 별도로 교대 측정한 생성 시간 감소율은 **19.32%**입니다. 각각 조합별 쌍별 감소율 중앙값의 중앙값이며, 이전 퍼센트를 더한 수치가 아닙니다. 모두 Mac M4 Max 측정입니다. [15차 재현·실패 원인·파형 보존](../records/2026-10-30/ROUND15_ENGINE.md)

## 호환 API

| 메서드·경로 | 용도 |
| --- | --- |
| `GET /v1/health` | 준비 상태, 샘플레이트, 8스텝·스레드 설정, 처리·대기 요청 수 |
| `GET /v1/styles` | 기본·가져온 음색 조회 |
| `POST /v1/styles/import` | 원본과 같은 JSON 또는 multipart `file` 형식으로 음색 벡터 가져오기 |
| `POST /v1/tts` | 원본 형식의 전체 음성 또는 선택한 스트리밍 응답 |
| `POST /v1/audio/speech` | `model`·`input`·`voice`를 사용하는 호환 별칭 |
| `POST /v1/tts/batch` | `items`·`defaults` 형식의 순차 배치, base64 오디오 목록 |
| `POST /v1/tts/stream` | `stream: true`와 같은 전용 경로 |
| `GET /v1/models` | 로드한 API 모델 ID 조회 |

`/v1/audio/speech`의 기본 모델 ID는 `supertonic-3`입니다. 실제 가중치는 `--model`로 선택한 F1200 계열이며 요청 중 모델을 교체하지 않습니다. `--model-id`로 API 이름을 바꿀 수 있습니다. 음색 가져오기는 이미 추출된 style JSON을 받으며 녹음 파일에서 음색을 추출하는 기능은 아닙니다.

### 전체 파일 응답

```bash
curl http://127.0.0.1:7788/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요. 음성을 생성합니다.","voice":"M1","lang":"ko","steps":8,"speed":1.0,"response_format":"wav"}' \
  --output speech.wav
```

```bash
curl http://127.0.0.1:7788/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"supertonic-3","input":"Hello from the optimized engine.","voice":"F1","lang":"en","response_format":"flac"}' \
  --output speech.flac
```

전체 응답은 `wav`(기본값), `flac`, `ogg`, `pcm`을 지원합니다. MP3·AAC·Opus로 요청하면 다른 형식으로 조용히 대체하지 않고 오류를 반환합니다. `X-Audio-Duration`, `X-Sample-Rate`, `X-RTF`, `X-Chunk-Count` 헤더로 결과를 확인할 수 있습니다.

## 스트리밍 방식

**텍스트를 문장 경계에 맞춰 묶고 각 묶음의 음성이 완성되는 즉시 보냅니다.** 첫 묶음을 보내기 전에 전체 문서의 음성을 만들지 않습니다. 각 묶음 안에서는 모델의 8회 denoising과 보코더를 끝까지 실행합니다. 단일 문장 내부를 실시간으로 조금씩 디코딩하는 causal 모델은 아닙니다.

| 요청 필드 | 기본값·동작 |
| --- | --- |
| `text` / `input` | 읽을 텍스트. 앞의 경로는 `text`, 호환 별칭은 `input` |
| `voice` | `M1`. `/v1/styles`에서 이름 확인 |
| `lang` | 원본 다국어 서버와 같은 `na` fallback. KO/EN 비교에는 `ko`·`en`을 명시 |
| `steps` | **8만 허용**. 7·50 등은 생성 전에 거부 |
| `speed` | 1.05. 범위 0.70~2.00, 소수점 두 자리까지(예: 1.25, 1.30). retime 패키지의 2.0에서만 내부 1.35 경로 사용 |
| `max_chunk_length` | 한국어 120자, 나머지 240자. 10~500자, 긴 무구두점 입력도 상한 적용 |
| `first_chunk_length` | 한국어 60자, 나머지 120자. 첫 묶음의 목표 길이. 완성 문장은 최대 길이 안에서 보존 |
| `silence_duration` | 묶음 사이 0.3초. 0으로 지정하면 추가 무음 없음 |
| `seed` | 42. 묶음마다 같은 seed로 시작해 재현 가능하게 생성 |
| `stream` | `false`. `true`이면 PCM 또는 SSE 응답 |
| `stream_format` | `audio` 또는 `sse`. 스트리밍일 때 사용 |
| `response_format` | 스트리밍은 `pcm`만 지원. 생략하면 자동 선택 |

문장 분할은 억양과 문장 간 리듬을 바꿀 수 있습니다. 스트리밍과 전체 파일을 샘플 단위로 대조하려면 **두 요청에 같은 chunk·seed·음색·배속 설정**을 사용하세요. 이 서버의 전체 응답도 같은 계획으로 묶음들을 합칩니다. 한 번에 전체 문장을 생성했던 기존 음성과 같다고 주장하지 않습니다.

### PCM 바이너리

```bash
curl --no-buffer http://127.0.0.1:7788/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"첫 문장입니다. 다음 문장을 듣는 동안 새 음성을 준비합니다.","voice":"M1","lang":"ko","steps":8,"speed":1.0,"response_format":"pcm"}' \
  --output speech.pcm
```

PCM은 **44,100Hz · 모노 · signed 16-bit little-endian**, 파일 헤더 없는 원시 오디오입니다. `X-Sample-Rate`·`X-Audio-Format`·`X-Audio-Channels` 헤더를 읽으세요. WAV 헤더를 중간마다 반복하지 않으며, 실제 음성 조각을 약 40ms 크기의 HTTP body 조각으로 나눠 보냅니다. 40ms는 전송 단위이며 첫 생성 지연을 뜻하지 않습니다. `stream:true`를 `/v1/audio/speech`에 보내도 같은 동작입니다.

### SSE

```bash
curl --no-buffer http://127.0.0.1:7788/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"첫 문장입니다. 이어서 다음 문장을 읽습니다.","lang":"ko","steps":8,"stream":true,"stream_format":"sse"}'
```

| 이벤트 | 데이터 |
| --- | --- |
| `start` | `sample_rate`, `channels`, `format`, `steps`, 전체 묶음 수, 서버 첫 오디오 준비 시간 |
| `audio` | 순서 `index`, `text`, `audio_base64`(PCM), `samples`, `duration_s`, `synthesis_seconds` |
| `end` | 합계 음성 길이·생성 시간·RTF·묶음 수. **정상 완료의 기준** |
| `error` | 스트리밍 시작 후 실패 코드. `end`를 보내지 않음 |

HTTP 200이나 `start` 수신은 생성 완료가 아닙니다. 첫 모델 묶음은 응답 헤더 전에 검사하므로 초기 오류는 JSON으로 반환합니다. 이후 실패는 SSE `error`, 바이너리는 연결 중단으로 전달하며 정상 종료와 혼동하지 않아야 합니다. 형식·음색·모델·8스텝 검증 오류는 `{ "error": { "message": "...", "type": "...", "code": "..." } }` 형태입니다.

브라우저 예제는 [web/streaming.html](../web/streaming.html)에 있습니다. 원시 PCM을 읽어 원래 샘플레이트로 Web Audio에 예약하고, 생성 완료 후 정상 WAV 헤더를 붙여 저장합니다. 완료 신호 없이 연결이 끊기면 완성 파일로 표시하지 않습니다.

사용자 청취에서 스트리밍 자체의 연결은 문제없었지만 1.5배속에서 일부 발음이 뭉개진다는 피드백이 있었습니다. 사용자가 처음 표현한 생략은 이후 정정됐으며, 단어 누락으로 확인된 것은 아닙니다. 보정 경로는 2.0에만 적용되므로 1.25·1.30 등 중간 배속은 기존 duration 경로를 시험하는 것입니다. 숫자 입력 지원을 발음 뭉개짐의 해결로 해석하지 않습니다.

## 지연과 재생 버퍼

처음부터 짧게 듣고 싶다면 `first_chunk_length:10`을 비교할 수 있습니다. 이 값은 첫 완성 문장부터 보내는 데 도움이 되지만 다음 문장 생성 시간이 길면 재생 버퍼가 비어 끊길 수 있습니다. 기본값은 더 긴 첫 묶음으로 여유를 둡니다. **전체 RTF가 1보다 작아도 무조건 끊김이 없다는 뜻은 아닙니다.** 네트워크·문장 길이·클라이언트 버퍼를 함께 확인해야 합니다.

맥 M4 Max·CPU 한 스레드·8스텝·선택 retime 패키지의 실제 로컬 TCP 검사에서, 기본 설정의 KO/EN·M1/F1·1×/2× 8조합을 각각 2회 측정했습니다.

| 항목 | 결과 |
| --- | --- |
| 조합별 첫 PCM 수신 시간 중앙값 | 약 0.15~0.43초 |
| 조합별 생성 RTF 중앙값 | 약 0.036~0.062 |
| 전체 응답과 스트리밍 PCM 대조 | 16회 모두 같음 |
| 문장당 실제 denoising 호출 검사 | 3묶음당 24회, 4조합의 두 응답 경로 모두 확인 |
| 취소 검사 | 216회 예정 요청을 8회 실행 뒤 중단, 다음 요청 정상 처리 |

RTF의 음성 길이는 설정한 묶음 사이 무음을 포함하며, 생성 시간은 모델 추론을 합산합니다. HTTP 전체 시간에는 인코딩·전송·수신 비용도 포함되므로 별도 `http_wall_rtf`로 기록했습니다. 재생 버퍼 검사는 SSE 도착 시각과 100ms 초기 여유를 사용한 시뮬레이션이며 실제 오디오 장치의 끊김 측정은 아닙니다. 이 수치는 **Mac의 표준 ORT 경로**이며 EPYC용 AVX2 v4나 ARM 성능이 아닙니다. 원시 결과: [기본 설정](../results/round11_mac_streaming.json), [짧은 첫 문장](../results/round11_mac_first_sentence.json), [실제 호출·취소 검사](../results/round11_streaming_runtime.json).

## 요청 처리와 운영 설정

하나의 전용 추론 worker만 사용합니다. 기본 대기는 4개이며 꽉 차면 `503 server_busy`와 `Retry-After:1`을 반환합니다. 상태 조회는 추론 때문에 막히지 않습니다. 연결을 끊으면 이후 묶음을 생성하지 않으며, 이미 실행 중인 ONNX 연산은 반환 시점까지 기다린 뒤 다음 연산 전에 취소합니다. CPU에서 실행 중인 C 연산을 강제로 중단하는 방식은 아닙니다.

```bash
python serve.py --model models/round8-latent-retime-g135 \
  --host 127.0.0.1 --port 7788 --max-pending 4 \
  --max-text-length 10000 --max-audio-seconds 180 --packet-ms 40
```

기본 bind는 loopback입니다. 브라우저 앱을 다른 origin에서 연결하면 `--cors http://localhost:3000`처럼 필요한 origin을 지정합니다. 가져온 음색의 기본 저장 위치는 Git 제외 경로 `.private/server-styles`입니다. 요청 본문은 2MiB, multipart style 파일은 1MiB로 제한하고, 긴 오디오와 배치 합계에도 길이 상한을 적용합니다.

## 검사 재실행

```bash
python -m pip install -r requirements-serve-test.txt
python tests_server.py
python tests_runtime.py
python experiments/check_streaming_runtime.py \
  --model models/round8-latent-retime-g135 --output results/streaming-runtime-new.json
# 위 serve.py를 실행한 상태에서 별도 터미널 사용
python experiments/benchmark_streaming.py \
  --output results/streaming-performance-new.json --repeats 2
```

성능 검사는 테스트 클라이언트의 버퍼링 시간을 쓰지 않고 실제 TCP에서 첫 `audio` 이벤트 수신까지 측정합니다. 새 출력 경로를 사용하며 비교 WAV는 Git에서 제외되는 `samples/`에 저장합니다. 기존 품질 한계와 소비한 개발 검증은 그대로 유지하며 이번 작업에서 새 WER·MOS 개선을 주장하지 않습니다.

참고: [원본 서버 문서](https://github.com/supertone-inc/supertonic-py/blob/main/docs/cli/serve.md), [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/), [Uvicorn 실행](https://www.uvicorn.org/deployment/).
