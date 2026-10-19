# Per_Platform_aug

Facebook, Instagram, KakaoTalk 이미지·영상 변형 측정 결과를 바탕으로 만든 **플랫폼별 Augmentation 정책**입니다. 이 `main` 브랜치는 기존 CSV 종속 전처리 코드를 대체합니다.

## 가장 먼저: 세 Python 파일의 역할

| 파일 | 무엇인가? | 언제 사용하나? |
|---|---|---|
| **[`augmentation_profiles.py`](./augmentation_profiles.py)** | 이미 완료된 측정 데이터로 사전에 구성한 **공통 변형 레시피 카탈로그**. 플랫폼·화질·해상도·방향에 따라 선택할 360개 결합 레시피 및 가중치를 정의 | **직접 실행하지 않음.** 아래 두 스크립트가 가져다 사용 |
| **[`offline_augment.py`](./offline_augment.py)** | 공통 레시피 중 하나를 선택하여 이미지/영상에 변형을 적용하고 결과 파일 및 manifest를 저장 | **학습 전에 미리 증강 데이터를 만들 때 실행** |
| **[`online_augment.py`](./online_augment.py)** | 똑같은 공통 레시피를 학습 중 선택·적용. 이미지: 메모리 내 PIL 변환, 영상: 필요 시 FFmpeg로 1회 임시 파일 재인코딩 | **학습 Dataset에서 호출** |

> 측정 CSV는 **이 코드를 만들 때 한 번 분석했으며, 실행 중 CSV를 다시 읽거나 새로 실측하지 않습니다.** 오프라인과 온라인의 정책은 동일합니다. 다만 변형 파일을 보존하느냐(오프라인), 학습 과정에서 바로 만드느냐(온라인)가 다릅니다.

## 설계와 근거

1. 초기 학습 혼합 비율은 **Clean 25% / Facebook 25% / Instagram 25% / KakaoTalk 25%**입니다. 사용자 트래픽 비율이 아니라 학습용 제안값입니다.
2. Facebook: 이미지 High/Mid/Low 각 1/3, 영상 High 50%·Mid 25%·Low 25%; Instagram은 3개 중복 화질을 통합한 `shared`; KakaoTalk: 이미지 Original/Normal/Low 각 1/3, 영상 단일 프로파일 `single`.
3. 입력의 모달리티, 해상도 밴드, 가로세로 방향 및 측정에 근접한 종횡비로 레시피 집합을 좁힙니다. **그 뒤 미리 정의한 레시피 하나를 선택**합니다. 각 레시피에는 원래 함께 관측된 Resize, Crop, JPEG QF 또는 영상 코덱/CRF/비트레이트, FPS, 밝기·대비가 묶여 있습니다.
4. **12개 프로파일 / 65개 조건 셀 / 360개 고정 결합 레시피**를 코드 내부에 보유합니다. Instagram 중복 화질은 중복 가중하지 않습니다. 레시피는 측정 행의 *결합된 관측값을 사전에 동결한 것*으로, 임의의 파라미터 조합을 새로 합성하지 않습니다.
5. 부가 Jitter(QF, CRF, 비트레이트, 광도, 크롭)는 기본 0입니다. 밝기·대비(`--p-photo=1` / `RuntimeConfig(p_photo=1.0)`)는 기본 활성화하며, **측정된 계수를 단일 패스에서 Y 채널에 적용**합니다.

## 오프라인 실행

```bash
pip install -r requirements.txt
# 영상은 시스템에 ffmpeg와 ffprobe, libx264, libx265 설치 필요
python offline_augment.py --input /path/to/train --output /path/to/train_aug --copies 3 --seed 42
```

`--copies 3`은 입력 하나당 3개의 독립적인 확률 샘플을 생성하며, `augmentation_manifest.jsonl`에 선택된 플랫폼·레시피 ID 등을 기록합니다. `--force-platform facebook --force-quality low`로 조건별 실험을 실행할 수 있습니다.

## 온라인 학습 적용: PyTorch Dataset

```python
from online_augment import RuntimeConfig, PlatformAwareOnlineImageAugmentor, OnlineImageDatasetWrapper

config = RuntimeConfig(seed=42)
aug = PlatformAwareOnlineImageAugmentor(config)
train_ds = OnlineImageDatasetWrapper(base_dataset, aug)

for epoch in range(num_epochs):
    train_ds.set_epoch(epoch)  # epoch에 따라 새로운 결정적 샘플링
    # DataLoader를 여기에서 생성하거나 worker에 epoch 값을 전달합니다.
    # training_loop(train_ds)
```

기본 wrapper는 `(PIL.Image, label)`, `(PIL.Image, label, key)`, 또는 `{'image': PIL.Image, 'label': ..., 'key': ...}`를 받습니다. 원본 Dataset이 이미 Tensor를 반환한다면 **Tensor 변환 전에** 증강하도록 조정하세요. 마스크가 있는 dict는 동일 Crop/Resize를 적용합니다. `persistent_workers=True`라면 각 worker의 epoch 상태도 갱신하거나 DataLoader를 재생성해야 합니다.

**영상**은 `PlatformAwareOnlineVideoAugmentor.augment_file(...)`를 사용하며, 실제 H.264/HEVC 인코딩 때문에 학습 속도가 느려질 수 있습니다. 임시 영상 파일 경로를 반환하는 방식이고, 사용 후 임시 파일을 삭제해야 합니다. 동영상 마스크 시간 정렬은 자동 지원하지 않습니다.

## 테스트

```bash
python -m unittest -v test_augmentation.py
python -m py_compile augmentation_profiles.py offline_augment.py online_augment.py
```

## 연구상 주의점

- 연구에서 사용한 480개 평가 표본을 학습 데이터에 포함하지 마세요.
- **실측 레시피의 파라미터를 적용하는 것과 플랫폼 다운로드 영상/이미지의 최종 SSIM·PSNR·비트레이트·밝기/대비가 같다는 것은 별개입니다.** 출력 재현성 검증은 독립적으로 수행해야 합니다.
- 밝기·대비는 이미 코덱 자체의 색 변화와 중복될 수 있습니다. 온라인/오프라인 실행 시 **실시간 재측정·반복 보정을 하지 않도록** 했으므로, 정밀한 포토메트릭 재현에 대해서는 사전 캘리브레이션을 별도로 수행해야 합니다.
- Instagram 중앙 크롭, 플랫폼별 확률, 입력 밴드 경계, 종횡비 허용폭 0.15, Lanczos 보간법, FFmpeg preset 등은 **구현상 가정 또는 학습 설계값**이지 플랫폼의 내부 정책으로 확인된 사실은 아닙니다.
- 이미지 `container_or_metadata_only`의 경우 픽셀 보존을 목적으로 파일을 복사하는 근사치입니다. 실제 메타데이터 제거와 바이트 재패키징은 구현하지 않았습니다.
- 측정 CSV 원본 및 연구용 이미지·영상은 공개 저장소에 포함하지 않았습니다. 정책 출처 파일들의 SHA-256은 `augmentation_profiles.py` 안에 기록되어 있습니다.