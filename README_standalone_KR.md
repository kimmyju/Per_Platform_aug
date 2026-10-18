# CSV 없이 실행하는 플랫폼별 Augmentation (Standalone)

이 스크립트는 실행 중 `measurements.csv`를 읽지 않습니다. 16개 측정 CSV에서 12개 **중복 없는** 플랫폼/모달리티/화질 프로파일을 만들고, 각 프로파일을 **해상도 밴드 × 가로/세로 방향 × 재인코딩 판정**별로 집계한 정적 정책을 Python 파일 안에 포함합니다. 360개 관측을 집계했습니다. Instagram High/Mid/Low의 중복 결과는 사진과 영상 각 1개 프로파일로 합쳤으며, 원본 480개 측정 쌍을 360개 서로 다른 정책 표본으로 줄여 사용합니다.

## 사용

```bash
pip install Pillow
# video: ffmpeg 및 ffprobe (libx264, libx265 포함)

python platform_aware_augment_standalone.py --show-profiles
python platform_aware_augment_standalone.py --input ./train --output ./train_aug --copies 3 --seed 42
python platform_aware_augment_standalone.py --input ./train --output ./fb_low --force-platform facebook --force-quality low
```

원본과 변환본의 쌍 추적은 출력 폴더의 `augmentation_manifest.jsonl`에서 확인합니다. 추후 학습에서 라벨은 원본과 동일하게 관리하되 크롭으로 조작 영역이 제거되는 경우는 주의하세요. 이미지 마스크는 `--mask-root`, `--mask-output` 옵션으로 동일 기하 변환을 적용합니다. 영상 마스크의 시공간 동기화는 별도로 구현해야 합니다.

## 구분해야 하는 확률

**측정 기반:** 입력 밴드와 방향이 동일한 표본 내 재인코딩 관측 개수에 비례한 확률(예: Facebook Mid 이미지 le1080 가로형에서 재압축 6/7, 보존 1/7). 이 확률은 앱 전체의 사용 빈도를 나타내지 않습니다.

**학습 설계안 (실측 발생 확률 아님):** `--p-clean 0.25 --p-facebook 0.25 --p-instagram 0.25 --p-kakao 0.25`. FB image 고/중/저 각각 1/3, FB video high/mid/low는 0.50/0.25/0.25, KT image original/normal/low 각 1/3. Instagram은 동일 집계값을 세 화질로 중복 가중하지 않음.

## 기본 설정: 임의 노이즈 미적용

`--jpeg-qf-jitter 0`, `--h264-crf-jitter 0`, `--hevc-bitrate-jitter 0`, `--ig-crop-jitter-prob 0`, `--photo-jitter 0`을 기본으로 합니다. QF/CRF, 출력 해상도, codec/FPS/GOP, 크로마, progressive 여부, 밝기 b와 대비 a의 **표본 집계 중앙값**은 Python 상수에 내장되어 있습니다. 앱 내부 인코더를 역추정하여 완전히 복제한 것이 아닙니다.

**밝기·대비:** 값과 실행 함수는 내장됐지만 기본 `--p-photo 0`입니다. 압축 자체에서 발생하는 밝기·대비 변화를 재현한 뒤 동일 변화량을 인코딩 전에 다시 더하면 *이중 적용* 위험이 있기 때문입니다. 연구자가 별도의 photometric ablation을 하려면 `--p-photo 1.0`을 명시하세요. 이 모드는 측정된 b/a를 인코딩 *전에 근사 적용*하므로 output b/a를 다시 측정하여 검증해야 합니다.

## 주의

1. 원본 자극(master)이 아니라 **플랫폼 업로드 입력과 같은 전처리 상태**의 학습 자료에 적용해야 합니다.
2. 12개 프로파일의 65개 밴드/방향 셀, 67개 판정 세부 그룹을 사용한 **집계 근사화**입니다. 실제로는 각 원본 자극의 종횡비에 대응한 결과가 달라질 수 있고, 본 코드는 그룹 중앙값을 사용합니다.
3. 측정에서 없었던 방향/밴드 조합은 함부로 만들어내지 않고 오류를 반환합니다. 반대로 입력 크기가 측정 범위를 크게 벗어나더라도 같은 밴드로 분류될 수 있으므로 사용자는 입력 밴드를 점검해야 합니다.
4. Identity / container-or-metadata-only는 원본 파일의 **바이트 복사**로 근사합니다. 이는 플랫폼 측의 메타데이터 제거까지 재현하지 않습니다.
5. Instagram center crop은 출력 정사각형으로부터 추정한 **구현상 가정**이며, 수동 드래그 여부까지 판정한 것이 아닙니다.
6. QF는 JPEG 양자화 테이블에서 추정, H.264 effective CRF는 추정값입니다. HEVC는 추정 H.264 CRF를 사용하지 않고 원본 비트레이트 × 실측 bitrate ratio 중앙값으로 설정합니다. 실제 생성 화질·비트레이트는 FFmpeg 설정과 입력 콘텐츠에 영향을 받습니다.
7. 실제 출력 SSIM / PSNR / 압축률 / 밝기 / 대비가 재다운로드 결과를 재현하는지는 **별도의 오프라인 평가**가 필요합니다. 코드가 CSV를 읽지 않는 것과 검증이 필요 없다는 것은 다른 문제입니다.
8. 학습에 쓰는 데이터만 처리하고, 성능 평가용 480개 표본을 학습에 섞지 마세요.
