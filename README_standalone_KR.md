# 플랫폼별 Augmentation — 실측 행 샘플링 v2

**학습 실행에는 CSV가 필요하지 않습니다.** 16개 측정 CSV에서 Instagram 중복 세 화질을 통합하여, 12개 프로파일·65개 밴드/방향 셀·360개 개별 실측 변형 조합을 Python 소스에 내장했습니다. 그룹 중앙값 하나를 반복하는 방식이 아니라 각 행의 크기·재압축 판정·QF/CRF·코덱·FPS·밝기 b·대비 a를 묶어서 샘플링합니다.

## 실행

```bash
pip install -r requirements.txt
# 영상에는 ffmpeg / ffprobe 및 libx264, libx265 필요
python platform_aware_augment_standalone.py --show-profiles
python platform_aware_augment_standalone.py --input ./train --output ./train_aug --copies 3 --seed 42
python test_standalone_empirical.py
```

## 정책과 확률

1. 플랫폼 확률 기본값: Clean 25%, Facebook 25%, Instagram 25%, KakaoTalk 25%. **실측 사용 비중이 아니라 학습 설계값**입니다.
2. Facebook 이미지 High/Mid/Low는 각 1/3, Facebook 영상은 50%/25%/25%, KakaoTalk 사진 Original/Normal/Low는 각 1/3. Instagram은 사진·영상별 공통 프로파일, KakaoTalk 영상은 단일 프로파일입니다.
3. 입력의 해상도 밴드와 가로/세로 방향을 구분하고, 관측된 종횡비와 비슷한 실측 행들에서 무작위로 하나를 선택합니다. 로그 종횡비 허용 오차 0.15, 근접 범위 ±0.025는 **코드상 설계값**이며 플랫폼 정책이 아닙니다. 관측되지 않은 조건은 오류로 처리합니다.
4. 선택된 실측 행의 기하·JPEG·영상 코덱·시간 설정·광도값을 결합된 하나의 정책으로 적용하고, 각 실행의 행 순번과 세부 파라미터를 `augmentation_manifest.jsonl`에 기록합니다.

## 밝기·대비 보정

기본 `--p-photo 1.0`으로 활성화되어 있습니다. 먼저 밝기·대비 추가 적용 없이 이미지를 JPEG 재저장하거나 영상을 H.264/HEVC로 재인코딩합니다. 그 과정에서 발생한 기울기 a·절편 b를 추정한 뒤, 목표 실측 a·b와의 **잔여 차이만** 보정해 다시 인코딩합니다. 오히려 오차가 커지면 보정본을 채택하지 않습니다. 영상은 필요 시 추가 시도 후 결과를 기록합니다.

**주의:** 내장 광도 추정기는 원래 포렌식 측정기의 displayed-Y 정밀 추정과 동일하지 않습니다. 영상의 기본 표본은 시작 4프레임이며, 영상 전체를 대표한다고 보장하지 않습니다. `--photometric-sample-frames 12`처럼 늘릴 수 있습니다. `--p-photo 0`으로 광도 보정 ablation을 실행할 수 있습니다.

추가 랜덤 편차(QF/CRF/비트레이트/Crop/광도 jitter)는 기본값이 모두 0입니다. Instagram 정사각형 Center Crop은 측정과 양립하는 **근사 가정**이며 실제 앱 정책으로 확인된 것은 아닙니다. 보존형 metadata-only 파일은 픽셀 보존 목적의 바이트 복사로 근사합니다. HEVC는 x264 추정 CRF를 x265 설정에 직접 사용하지 않습니다.

## 연구 검증 한계

- 480개 평가용 표본 또는 그 파생 데이터는 학습에 포함하지 마세요.
- 입력은 플랫폼 업로드 직전 stimuli와 동등한 상태여야 측정값의 의미가 유지됩니다.
- 이미지 마스크에 대한 동기 기하 변환을 지원하며 **영상 마스크 시간 정렬은 구현되지 않았습니다**.
- 코드가 생성한 출력의 실제 SSIM, PSNR, bpp/bitrate ratio, 밝기·대비는 별도 측정이 필요합니다. 샘플 실행 성공은 플랫폼 재다운로드와 동등한 출력 품질을 증명하지 않습니다.
- 공개 저장소에는 원본 측정 CSV를 게시하지 않았으며, SHA-256 출처만 `standalone_policy_provenance.json`에 기록했습니다.
