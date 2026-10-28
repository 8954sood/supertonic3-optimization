# 3차 실험 — CFG 교사 출력을 단일 경로로 증류

2차의 안전한 비교점은 `models/mixed-v1-selected`로 고정 보존한다. 아래 실험은 **원래 CFG4와 수학적으로 같은 계산이 아니다**. 원본CFG4의 업데이트를 한 조건경로가 학습해 근사하는 별도학생이다. 실제 denoising은8회 유지한다. 성공하면 단계당 두조건경로를 한경로로 줄여 계산량을 낮출 수 있지만, 음질·가이던스 일치 검증이 필요하다.

- 원본FP32모델을 conditional graph로 변환: `models/cfg-student-fp32-source`. 초기가중치만 복사해 단순CFG1로 실행한 것을 완성모델로 취급하지 않는다.
- 미분가능한 단일경로 구현을 그 경로의 ORT출력과 대조한 결과: `results/torch_cfg_graph_parity_v1.json`.
- 초기학생은 혼합폭초기가중치. 원본CFG4의 실제8스텝1600훈련쌍과24검증단계쌍을 사용. 입력은교사궤적이며 원본CFG4다음latent를목표로함.
- 벡터장의모든학습가중치를학습. 2400회·LR3e-5·200회마다검증. checkpoint `distill/joint-cfg-mixed-v1`, 로그 `results/train_joint_cfg_mixed_v1.log`.
- 이후 실제학생8스텝 rollout과 음향/ASR/서버속도를 확인한다. 기존test12문장은 이미소비했으므로 이실험의 독립시험으로 재사용하지 않는다. 후보선택은기존validation만사용하고 독립시험은새텍스트로분리한다.

## 중간 결과

단일 경로 구현의 ORT 수치 대조는 상대RMSE6.87e-8/최대오차4.77e-7로 통과했다. 교사CFG4와 같다는 검사가 아니다.

2400회 학습 중 600회 시점 모델을 별도 ONNX로 보존했다. validation4문장 logmel차이는 평균 약3.50dB, F0문장별중앙오차 평균25cents, KO2/EN2 ASR오류0이었다. 2차음질우선후보의2.754dB보다 차이가 크므로 실제서버속도를 함께 측정한다. 원시: results/acoustic_cfg_mixed_preview_v1.json, results/asr_cfg_mixed_preview_v1.json. 기존test12는 재사용하지 않는다.

## 서버 중간 측정과 추가 학습

600회 단일경로 혼합폭 preview를 실제서버에서 CPU1·8스텝으로 워밍업후각1회 측정: KO6.006초/EN5.418초, 표본최대RSS394.3MB. 초기탐색용단발측정이며 최종3회이상측정을대체하지않는다. 내부guidance는단일경로이며 교사CFG4를학습한근사값이다. 원본CFG4동일계산이라고표현하면안된다. 원시 `results/epyc_cfg_mixed_preview_v1_8.json`, `results/guard_cfg_mixed_preview_v1.json`.

2400회 공동학습최종은 교사update검증MSE0.11213→0.04219로 감소했다. 그러나 validation4문장 음향은 최종보다600회preview가가까웠다(최종logmel평균약3.757dB/ASR오류0, preview약3.503dB/오류0). checkpoint선택은손실하나로판단하지않는다.

### 전체8스텝 음성 손실

`distill/rollout-cfg-mixed-v1`: 위600회체크포인트에서 시작, 실제학생8회연결+고정원본FP32보코더를통해 latent상대MSE와MRSTFT로학습. LR3e-6,200회,200음색/seed조합,검증8문장. 100회까지검증손실0.62819→0.35109, 최종latent상대MSE0.42044→0.19278로줄었다. 아직ONNX실제음향/발음확인전이므로채택하지않는다.

### 더 작은 반복부

`distill/joint-cfg-1024-v1`: 혼합폭대신34.6M수준의1024학생을같은단일경로/교사CFG4목표로2400회학습중. 데이터/split은동일. 후보를빠르게만들기위한실험이며음질검증전기본값변경없음.

### FP32 보코더 축소 시도

- `distill/collect_decoder.py`: 원본교사final latent200조합에서10개FFN입출력수집, 문장당128행. 훈련25600행/검증1024행. 데이터 `distill/decoder-data-v1`, 원본decoderprobe.
- `distill/decoder1024-v1`: decoder10개FFN을2048→1024로ridge+1600회층별학습, QAT없음. 일부층검증오차가큼.
- `distill/decoder-joint1024-v1`: 위모델의모든decoder학습가중치를원본오디오의정렬파형MSE+MRSTFT로800회추가학습. 초기검증wave상대MSE0.37370→0.27609, MRSTFT0.84908→0.70127로감소했지만원본파형재현차이가여전히커미채택. 새보코더를현재추론패키지에넣지않았다.
- 새harmonic/noiseDSP헤드나적대적훈련은아직아님. 원본구조의FP32폭축소실험이다.

## 지원스텝과 품질기준

한경로학생은 현재8스텝으로만훈련/검증했다. manifest.supported_steps=[8]을기록하고실행기가이목록을따른다. 모든모델에서8미만은기존처럼거부하고, 원본모델의기존범위는유지한다. tests_runtime.py의원본공식core동등성및오류검사가다시통과했다(results/tests_runtime_round3.log). 새학생에서7/12요청거부도확인했다.

FP32보코더반폭공동학습의최선은800회였지만검증wave상대MSE0.2761로원본파형재현기준에미달해미채택했다. decoder양자화나축소를새선택모델에적용하지않는다.

## 후보를 고정한 뒤의 검증

- 단일경로혼합폭600회preview가4문장음향/ASR에서가장나아 `models/cfg-mixed-v1-selected`로보존했다. source checkpoint는별도 `best_step_00600.pt`로남아있다.
- 2400회최종과200회음성rollout은교사latent손실이줄어도실제validation4문장logmel이각3.757/3.611dB로preview3.503dB보다나아지지않았다. 파형·latent손실감소만으로품질개선을선언하지않았다.
- 혼합폭단일경로의확대validation8문장: KO2/88(원본2/88), EN0/46(원본0/46). 이결과로후보를고정했다.
- 새독립텍스트 `distill/corpus_cfg_v3.json` 12문장으로평가: 원본KO5/139, EN0/69; 단일경로혼합폭KO5/139, EN3/69. 따라서영어WER4.35%가남는다. 청취동등성은미확인이다. 기존test12는재사용하지않았다.
- 1024단일경로2400회도종료. validation4문장ASR오류0이나logmel평균약4.564dB로혼합폭보다차이가크고, 확대validation8문장은KO4/88(원본2/88), EN0/46으로한국어발음도악화했다. 기본품질후보로대체하지않는다. 속도·품질절충을기록하기위해실제서버속도도측정한다.
- test v3의같은12문장으로학습이이미끝나고고정된2차CFG4후보와1024단일경로도추가비교한다. 이비교를위해새가중치를훈련하거나조정하지않는다. 시험결과를앞으로학습/후보선택에재사용하면독립성이없다.

## 사용자 청취 피드백

사용자가원본과다음실제서버샘플을비교한뒤 **“이제 거의 차이 없는듯”**이라고답했다. 비교대상은M1한국어benchmark문장, seed42,8스텝이었다.

- 원본: `samples/equivalence_fp32_cached/ko_short_reference.wav`
- 빠른후보: `samples/epyc_cfg_mixed_preview_v1_8/ko_short_8steps.wav`
- 모델: 600회 단일경로혼합폭preview, 현재 `models/cfg-mixed-v1-selected`와같은가중치.

이샘플을현재음질기준으로보존한다. 이것은해당샘플에대한사용자의직접청취의견이며, 모든음색·언어·장문에대한동등성이나MOS평가를뜻하지않는다. 특히독립시험의EN3/69오류는그대로남긴다. 1024더작은후보로자동대체하지않는다.


## 마무리 확정

전체 학습과 서버 측정은 종료했다. 선택 후보 KO5.823초/EN5.458초·394MB, 더 작은1024 KO4.707초/EN4.392초·388MB(각3회중앙값). 새시험1024 KO8/139·EN0/69, CFG4혼합폭 KO5/139·EN0/69. 선택 후보의12문장 평균logmel차이는3.943dB. 청취 기준은600회혼합폭으로유지한다. 최신 확정 상태는 [HANDOFF_ROUND3.md](HANDOFF_ROUND3.md)를 따른다. 위 본문의 학습중/미확인 표현은 당시 시점 기록이다.
