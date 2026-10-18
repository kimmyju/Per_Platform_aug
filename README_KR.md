# 플랫폼 변형 기반 Augmentation 실행 안내

## 1. 구성
- `platform_aware_augment.py`: 이미지 및 영상 전처리 스크립트
- `measurements/`: **16개 개별 측정 CSV 원본** (공개 저장소에는 미포함; 로컬에서 지정)
- 모델 훈련에 사용하는 **업로드 직전의 자극(stimuli)과 같은 처리 상태의 학습 데이터**를 입력으로 사용해야 합니다. 연구에서 지표를 계산한 기준은 original 마스터가 아니라 `stimuli` 업로드 파일입니다.

## 2. 요구 환경
```bash
pip install Pillow
ffmpeg -version
ffprobe -version
# ffmpeg에는 libx264, libx265가 있어야 함
ffmpeg -hide_banner -encoders | grep -E 'libx264|libx265'
```

## 3. 실행
```bash
python platform_aware_augment.py \
  --input /path/to/TRAIN_uploaded_stimuli \
  --output /path/to/TRAIN_augmented \
  --measurements-dir ./measurements \
  --copies 3 --seed 42 --epoch 0
```

- 입력 폴더를 재귀적으로 스캔해 이미지와 비디오를 모두 처리합니다.
- 이미지·영상이 공존해도 이미지/영상 별개의 측정 프로파일을 자동 사용합니다.
- `--copies 3`은 입력 1개당 서로 독립적으로 샘플링한 3개 변환본을 만듭니다. 데이터를 train/validation/test로 나눌 때는 **복제 전 원본 기준으로 분리**해야 합니다.
- `--epoch 1`처럼 바꾸면 동일 seed에서도 다음 에폭용 변형본을 생성할 수 있습니다.
- 출력의 `augmentation_manifest.jsonl`에 원본, 플랫폼, 품질, 샘플링한 stimulus_id, 적용 함수/수치, 최종 출력 FPS/코덱/비트레이트 비율이 기록됩니다.

### 이미지 세그멘테이션 마스크를 함께 처리
```bash
python platform_aware_augment.py \
  --input train/images --output augmented/images \
  --mask-root train/masks --mask-output augmented/masks \
  --measurements-dir ./measurements
```
마스크는 이미지와 동일한 Crop/Resize가 적용되며 **Nearest Neighbor**로 보간하고 **PNG**로 저장합니다. 이미지와 동명 `.png` 파일이 마스크 입력입니다. 영상 마스크는 자동 변환되지 않으므로 manifest의 동일 기하·시간 정보를 사용하여 후처리해야 합니다.

### 정책 내용만 확인
```bash
python platform_aware_augment.py --measurements-dir ./measurements --show-profiles
```

### 강제 실험 / Ablation
```bash
python platform_aware_augment.py --input train/images --output out_FB_low \
  --measurements-dir ./measurements --force-platform facebook --force-quality low
```

## 4. 확률(초기 학습 제안, 플랫폼 관측 빈도 아님)
| 1차 선택 | 확률 | 내부 품질 선택 |
|---|---:|---|
| Clean | 25% | 추가 변형 없는 원본 파일 복사 |
| Facebook | 25% | 이미지 high/mid/low 각 1/3; 영상 high 50%, mid 25%, low 25% |
| Instagram | 25% | 이미지·영상 각각 3화질의 일치하는 측정결과 1개로 통합 |
| KakaoTalk | 25% | 이미지 original/normal/low 각 1/3; 영상 단일 1개 |

플랫폼/품질을 선택한 뒤에는 **입력 해상도 밴드·종횡비가 유사한 측정 행** 중 1개를 샘플링합니다. `container_or_metadata_only`이면 픽셀 변형하지 않고 **파일 복사**, 나머지는 선택된 사례의 변환 비율·코덱/압축 설정을 적용합니다.

### 주요 변환과 확률
| 기능 | 확률 / 매개변수 | 근거와 한계 |
|---|---|---|
| 이미지 재인코딩 | 조건부 실제 측정 행별 판정 | JPEG QF 참고값: FB 93, IG 92, KT Normal 96, KT Low 86; 탐색 ±1 QF(제안) |
| 이미지 색차 변환 | 재인코딩한 사진 100% | 측정 행의 4:2:0/4:4:4, Progressive True/False 반영 |
| 이미지 Resize | 선택된 실측 `down_width/orig_width`, `down_height/orig_height` | 임의 공통 scale 대신 밴드·종횡비 조건부 |
| IG 이미지 Crop | 해당 프로파일 100% | 기본 Center Square; 위치 흔들림 확률 10%, 여유 크롭 마진의 ±3%(제안) |
| 영상 재인코딩 | 변환 프로파일이면 100% | FB/IG H.264, KT 1080/2160p HEVC·720p H.264 |
| 영상 추정 CRF | H.264: 선택 행 `effective_crf_interp` ±0.6(제안) | HEVC에는 x264 추정 CRF를 직접 쓰지 않음 |
| 영상 비트레이트 | HEVC: 원본 bitrate × 선택 행의 실측 `bitrate_ratio` ±8%(제안) | 1-pass ABR의 출력은 목표에 완전히 일치하지 않을 수 있으며 manifest에서 검증 |
| GOP / B-frame | 변환된 영상 100% | 선택 행의 `down_gop_mean`, `down_has_b_frames`를 재현 |
| FPS / 프레임 오프셋 | 변환된 영상 100% | 선택 행의 `down_avg_frame_rate`, `temporal_offset_frames` 적용. KT 720p는 주로 24fps |
| 밝기·대비 | 재인코딩되는 샘플에 각각 함께 적용할 확률 `--p-photo=1.0` | 실측 행의 b와 a, 각각 편차의 ±10% 변동(제안). Clean/픽셀 보존형은 미적용 |

### 값 변경 예시
```bash
python platform_aware_augment.py --input train --output aug \
 --measurements-dir ./measurements --copies 2 \
 --p-clean .30 --p-facebook .30 --p-instagram .20 --p-kakao .20 \
 --jpeg-qf-jitter 1 --h264-crf-jitter 0.6 --hevc-bitrate-jitter .08 \
 --p-photo 1.0 --photo-jitter .1 \
 --ig-crop-jitter-prob .1 --ig-crop-jitter-frac .03
```

## 5. 확인 및 제약
- 이번 측정 표본은 각 밴드 10개로 작아 **개별 플랫폼의 전체 사용자 변환 확률**을 의미하지 않습니다. 실측 기반 경험적 재현용입니다.
- `Crop` 위치는 정렬 shift만으로 center라고 확정할 수 없어서 중앙 근사치로 구현했습니다.
- 원본 파일 해시가 없으면 동일한 이미지 파일이라는 사실을 CSV만으로 새로 검증할 수 없지만, IG high/mid/low의 주요 변환 필드 일치를 16개 CSV로 확인하고 프로파일 중복 가중치를 막았습니다.
- 색차/양자화/색공간·영상 codec 구현 차이로 출력이 재다운로드된 파일과 **바이트 단위로 같아지지는 않습니다**. 특히 HEVC 비트레이트 비율은 1-pass 인코더가 달성하지 못할 수 있어 manifest의 실제 출력 값을 확인하고 파라미터를 캘리브레이션해야 합니다.
- 밝기·대비 보정은 **인코딩 전 근사치**로, 코덱 자체가 만든 밝기 변화와 중첩될 수 있습니다. 최종 강건성 실험 전 일부 파일에 대해 b/a를 재측정하여 확인하세요.
- Clean 성능과 각 플랫폼(해상도 밴드별) AUC/BACC/Fake Recall을 함께 측정하고 정책 확률을 재조정하세요.