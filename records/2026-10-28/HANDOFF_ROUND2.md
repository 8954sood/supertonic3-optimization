> 당시 시점의 기록입니다. 최신 확정 상태는 [HANDOFF_ROUND3.md](HANDOFF_ROUND3.md)를 참조하세요.

# 2차 작업 인계 — 이 문서를 먼저 읽기

2026-10-28. 앞선 HANDOFF.md의 상태는 1차 기록이다. 최신 결과는 ROUND2.md와 round2_metrics.json을 따른다.

## 현재 상태

현재 후보는 `models/mixed-v1-selected`, CPU1·CFG4·실제8스텝. 서버9.995초KO/9.096초EN, 표본RSS356.9MB. 작은1024모델보다 느리지만 실제서버 음향이 원본에 가까워졌다. 원본과의 청취동등성은 미확인이고 실시간 목표도 미달. 원본모델을 덮어쓰지 않았다.

선택한 것은 mixed 공동학습의 **400회 시점에 내보낸 ONNX**다. `distill/joint-mixed-v1/best.pt`는 현재1200회이며 선택모델과 다른가중치다. 400회FP32체크포인트는 별도저장되지 않았다. 선택ONNX는 preview에서 보존했고 manifest의vector_sha256로 식별한다. 이후train_joint/train_rollout은 best_step_XXXXX.pt도 보존하게 수정했다. 후속FP32훈련은 해당400회라고 가정하지 말고 원래init부터 재현하거나 다른체크포인트를 명시해 시작한다.

훈련은 모두종료. 서버벤치마크도종료. 마지막 정리의 음향분석/패키징이 남아있는지는 해당로그/프로세스 확인. 계속실행중인 학습이 있다고 가정하지 않는다. 서버는 `<REMOTE_WORKSPACE>`, 기존가상환경을 유지했다. 인증개인키를 열거나 패키지에 넣지 않는다. 서비스나 다른프로그램 변경없음.

## 수행한 학습

- `distill/torch_graph.py`: 원본ONNX 반복부·보코더의 미분가능한PyTorch훈련구현. 원본ORT parity12입력검사 통과. 운영추론은 여전히ORT CPU.
- `distill/trajectories-v1`: 고유훈련40텍스트×1음색/seed=40조합320단계쌍, 검증8텍스트64쌍.
- `distill/trajectories-v2`: 같은40텍스트×5음색/seed=200조합1600단계쌍, 검증동일. 새텍스트200개가 아니다. 약289MB.
- `distill/joint1024-v1`: FFN공동학습600회. 음향오차조금감소, 여전히큼.
- `distill/rollout1024-v1`: 전체8스텝+고정보코더+latent/MRSTFT200회. 일부발음악화. 제외.
- `distill/joint1536-v1`: FFN공동학습1000회. 1024보다오차감소.
- `distill/rollout1536-v1`: 위모델에서200회 전체음성학습. 최선step0으로개선없음.
- `distill/rollout1536-v2`: 200조합으로400회추가. 역시최선step0. 개선한것처럼기록하지않기.
- `distill/mixed-v1/selection.json`: 512폭2층·1024폭4층·1536폭12층·원본2048폭10층. 초기체크포인트initial.pt.
- `distill/joint-mixed-v1`: 1600단계쌍으로1200회공동학습. update검증MSE최선은1200회이나 실제음향/ASR는400회내보내기가더좋아400회선택.
- 보코더QAT는 fake/실제정수출력의불일치0.222 상대RMSE 때문에훈련시작안함. fakequant함수를정확한ORT실행기로간주하지말기.

## 모델·결과

- selected: 현재후보. frontend/decoder는FP32 MatMul, 반복부는reduce_range INT8. 원본LICENSE유지.
- mixed-v1-quality/fast:1200회모델. 400회보다검증음질/ASR나빠선택안함.
- mixed-v1-cached: 동일400회가중치의조건캐시. Mac4조합비트일치지만서버11.2/11.9초·382MB로악화해제외.
- round2_metrics.json: 추가실험원시파일해시·음향·ASR·학습보고서모음.
- runtime패키지: dist/mixed-v1-quality-runtime.tar.gz 및.sha256. Python>=3.12, requirements설치후synthesize.py --model model.

## 시험 세트 주의

corpus.json의test12문장을 처음사용했다. 원본KO4/142, EN0/72; selected KO4/142, EN1/72. 이후test결과로튜닝하지않았다. 이제이12문장은이미공개된시험결과이므로 추가후보선택에반복사용하면독립성이없다. 다음튜닝에는새holdout이필요하다. validation8문장은계속개발용으로쓸수있다.

## 다음에 우선할 일

1. 실제서버새후보와원본의A/B청취: `samples/epyc_mixed_selected_v1_8/ko_short_8steps.wav` / `samples/equivalence_fp32_cached/ko_short_reference.wav`. 같은benchmark텍스트·M1·seed42·8회. 새후보도동등성미확인.
2. **CPU출력안정성**: 동일모델의Mac/EPYC를비교했더니초기전체INT8의길이가KO3.18%/EN8.13%달랐다. 4모듈해시일치확인. 원인후보는AVX2 U8S8포화지만, 단독원인은미확정. 이미있는fast-avx2-int8(reduce_range)과U8U8/fullrange대조를각모듈별로수행. Mac음성만듣고구형x86품질을통과시키지말기.
3. 현재반복부mixed51M수준이므로목표8M증류에는아직멀다. 고유텍스트40개뿐인훈련을확대하고언어·음색·길이·seed를균형화. 더작게줄이기전에실제타깃음질확인.
4. 원본FP32frontend/decoder를유지한채파라미터축소·저장정밀도·커널최적화를분리. 현재decoder는새DSP헤드가아니다. 새harmonic/noise헤드는별도latents→audio모델과대규모교사오디오·스펙트럼/배음/위상훈련이필요하다.
5. 실제서버벤치마크는nice19·affinity0·8회·guard420MB/여유80MB. 현시스템에는memory cgroup이없으므로감시기를하드제한으로표현하지않기. ARM실측은여전히없다.
6. 수치정의와데이터split을반드시붙이기. waveform상관·logmel dB·CER/WER·화자/청취유사도를한개의일치율로합치지않기.
