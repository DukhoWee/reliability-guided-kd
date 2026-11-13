# Reliability-Guided Knowledge Distillation

CIFAR-100-C의 손상 이미지에서 Student를 학습하기 위한 핵심 Knowledge Distillation 구현입니다. 기존 실험 프로젝트에서 최종 MSP+JSD 학습 경로와 필요한 데이터·모델 유틸만 추출했습니다.

## 핵심 방법

- **Teacher reliability**: noisy posterior의 MSP uncertainty와 clean/noisy posterior의 정규화 Jensen–Shannon divergence를 결합합니다.
- **Adaptive target blending**: 샘플별 위험 점수에 따라 clean posterior의 혼합 비율을 조절합니다.
- **Knowledge distillation**: noisy 이미지만 입력받는 Student를 CE와 temperature-scaled KL loss로 학습합니다.

```text
p_clean = softmax(clean_teacher(clean_image))
p_noisy = softmax(noise_teacher(noisy_image))
score = (1 - max(p_noisy)) + JSD(p_clean, p_noisy) / log(2)
w_clean = 0.5 * clamp(score, 0, 1)
target = (1-w_clean) * softmax(noisy_logits/2) + w_clean * softmax(clean_logits/2)
loss = 0.2 * CE(student(noisy_image), label) + 4 * KL(target || softmax(student_logits/2))
```

Teacher는 각각 별도로 로드한 ResNet-50이며, Student는 ResNet-18입니다. clean Teacher와 corruption fine-tuning한 noise Teacher의 checkpoint 출처가 다릅니다. 위 수식의 계수는 아래 실행 명령으로 지정하며, 학습 스크립트의 모든 기본값이 최종 실험 설정인 것은 아닙니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `scripts/train.py` | JSD/MSP reliability, adaptive target, Student 학습, calibration, validation, checkpoint, corruption 평가 |
| `scripts/kd_data.py` | clean/noisy Dataset, image ID 기준 split, checkpoint 로딩·저장 보조 함수 |
| `scripts/mcdo_utils.py` | CIFAR용 ResNet, 정규화, 데이터 읽기 및 평가 지표 |
| `SOURCE_MANIFEST.json` | 원본 파일·해시와 추출 변경 사항 |

`train.py`의 `reliability_components`, `reliability_score`, `train_one_epoch`, `kd_kl_with_probs_per_sample`이 핵심입니다. Entropy/JSD/MSP ablation과 hard switch, selective blend, fixed blend도 같은 학습 함수에서 선택할 수 있습니다. Soft blend의 target은 quantile threshold와 무관하며, threshold mask는 이 모드에서 통계용으로만 사용합니다.

## 실행

Python 3.10 이상 환경에서 실행합니다. GPU 환경에서는 해당 CUDA 환경에 맞는 PyTorch를 먼저 준비하세요.

```bash
pip install -r requirements.txt
python scripts/train.py --help
```

별도로 준비할 파일:

- `cifar-100-python.tar.gz`와 CIFAR-100-C의 corruption `.npy` 및 `labels.npy`
- `models/edadaltocg/resnet50_cifar100/`: 원본 실험에서 사용한 Teacher `config.json`과 지원 형식의 가중치
- `models/edadaltocg/resnet18_cifar100/`: Student `config.json`과 초기 가중치
- corruption fine-tuning한 ResNet-50 noise Teacher checkpoint

모델 구조는 CIFAR용 conv1(3×3, stride 1), maxpool 제거 구조입니다. 임의의 ImageNet checkpoint와 호환된다고 가정하면 안 됩니다. 정확한 파일 로딩 규칙은 `load_teacher_state_dict`와 `load_checkpoint_state_dict`에 있습니다. 데이터와 모델은 저장소에 포함하지 않습니다.

```bash
python scripts/train.py \
  --noise-teacher-checkpoint /path/to/noise_teacher.pt \
  --gate-signal msp_js \
  --gate-mode soft_blend \
  --alpha-u 1.0 --beta-js 1.0 --gamma-disagree 1.0 \
  --fallback-clean-weight 0.5 --fallback-eta 0.0 \
  --tau 2.0 --alpha-ce 0.2 --alpha-kd 1.0 \
  --epochs 20 --seed 42 --split-seed 42 \
  --output-dir checkpoints/msp_js_soft_blend
```

clean Teacher는 기본 model-dir에서 로드하며 `--clean-teacher-checkpoint`로 바꿀 수 있습니다. 원본 실험의 다른 seed는 13, 21, 87, 101입니다. 이 저장소에는 다중 seed orchestration과 결과 보고 도구를 포함하지 않았습니다.

## 데이터 분할 및 검증 범위

공식 CIFAR-100 test 이미지 10,000장과 대응되는 CIFAR-100-C를 원본 이미지 ID 기준으로 calibration/train/validation/test로 나눕니다. 기본 크기는 500/7600/950/950이며, 동일 원본 이미지의 corruption 변형이 split을 넘지 않도록 합니다. 공식 CIFAR train 전체에서 Student를 학습하는 구성은 아닙니다. 기본 checkpoint는 validation noisy NLL 최소값으로 선택합니다.

추출 후 CLI 정상 실행과 작은 CPU batch에서 원본과의 신호·학습 update 동등성을 검증했습니다. MSP+JSD/Entropy+JSD × 4개 혼합 모드의 8가지 조합에서 loss 통계와 optimizer update 후 파라미터가 정확히 일치했습니다. 전체 CIFAR 학습을 새로 실행하여 성능을 재현한 저장소는 아닙니다.

## 코드 출처

최종 학습 코드는 원본 `scripts/train_multi_teacher_gated_kd.py`에서 가져왔고, 공통 데이터 함수는 `train_noisy_student_kd.py`에서 추출했습니다. 초기 CDF 학습 루프, 분석·시각화 코드, 실행 산출물과 대용량 파일은 제외했습니다. 학습 알고리즘의 계산식은 변경하지 않았습니다.

원본 HEAD는 `778311c935f499ad2075dddc7fa20f30f59c0f20`이며 최종 multi-teacher 학습 파일은 당시 Git 미추적 상태였습니다. 추출 기록은 `SOURCE_MANIFEST.json`에 있습니다. 이 새 저장소의 최초 커밋은 추출본을 보관한 이력이며 원본 알고리즘의 최초 작성 시점이나 단독 작성자를 입증하지 않습니다.
