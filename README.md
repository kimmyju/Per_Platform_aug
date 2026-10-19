# Per_Platform_aug — Standalone v2 (CSV 없이 실행)

Facebook·Instagram·KakaoTalk에서 관측한 이미지·영상 변환을 학습 데이터에 근사 적용하는 **오프라인 전처리 프로젝트**입니다. 이 브랜치에는 **예전 CSV 기반 버전과 새 Standalone 버전이 함께 남아 있습니다.** 실제 실행할 파일은 목적에 따라 하나만 선택하세요.

## 가장 먼저: 각 파일이 하는 일

| 파일 | 역할 | 직접 실행? |
|---|---|---|
| **[`platform_aware_augment_standalone.py`](./platform_aware_augment_standalone.py)** | **최신 버전(v2).** 실행 중 CSV 없이, 내장된 360개 실측 변형 조합을 조건별 샘플링해 이미지·영상 augmentation 수행. 밝기·대비 잔차 보정 기본 활성화 | **예 — 보통 이 파일을 실행** |
| [`platform_aware_augment.py`](./platform_aware_augment.py) | **이전 CSV 기반 버전.** 실행 시 16개 `measurements.csv`를 `--measurements-dir`로 제공해야 함. 새 Standalone의 실행에 필요하지 않음 | 선택 사항 — CSV 방식 비교 시에만 |
| [`test_standalone_empirical.py`](./test_standalone_empirical.py) | Standalone에 내장된 프로파일 수·행 샘플링·동일 seed 재현성 등을 확인하는 **단위 테스트**. 모델 학습 코드는 아님 | 최초 설치/코드 변경 후 테스트 |
| [`README_standalone_KR.md`](./README_standalone_KR.md) | **최신 Standalone v2의 한국어 설명서**: 정책, 확률, 광도 보정, 검증 한계 | 읽는 문서 |
| [`README_KR.md`](./README_KR.md) | **이전 CSV 기반 코드의 한국어 설명서**. 최신 Standalone 설명서가 아님 | CSV 버전 사용할 때 참고 |
| [`standalone_policy_provenance.json`](./standalone_policy_provenance.json) | 내장 정책 생성에 사용된 측정 CSV들의 **SHA-256 출처/무결성 기록**. 변환 함수나 프로파일 데이터 파일이 아니며 실행 중 읽지 않음 | 실행 불필요 |
| [`requirements.txt`](./requirements.txt) | Python 의존성(Pillow, NumPy) 설치용 | `pip install -r ...` |
| `.gitignore` | Git에서 모델 가중치·측정 CSV·생성 데이터 등이 실수로 추적되지 않도록 제외 | 실행 불필요 |
| `README.md` | 지금 보는 **파일 안내 및 시작점** | 읽는 문서 |

> **한 줄 요약:** CSV 없이 최신 증강을 돌리려면 **`platform_aware_augment_standalone.py`만 실행**하세요. `platform_aware_augment.py`는 이전 버전을 보관한 파일이므로 둘을 순서대로 실행하지 않습니다.

## Standalone v2 빠른 실행

```bash
git clone -b standalone-csv-free https://github.com/kimmyju/Per_Platform_aug.git
cd Per_Platform_aug
pip install -r requirements.txt
# 영상은 별도로 FFmpeg / FFprobe와 libx264, libx265 설치 필요
python platform_aware_augment_standalone.py --show-profiles
python test_standalone_empirical.py
python platform_aware_augment_standalone.py --input /path/to/train --output /path/to/augmented --copies 3 --seed 42
```

- `--input`은 **학습 데이터** 폴더, `--output`은 증강 결과 폴더입니다. **평가용 480개 표본을 학습에 넣지 마세요.**
- `--copies 3`은 입력 1개당 변환본 3개를 만드는 옵션입니다. 전처리 코드이므로 모델 학습 자체는 수행하지 않습니다.
- 기본 학습 혼합 비율 **Clean/FB/IG/KT = 25%씩**은 *측정된 실제 사용자 비율이 아니라 실험 설계값*입니다.
- JPEG QF/CRF/비트레이트·Crop 추가 jitter는 기본 **0**이고, 밝기·대비 보정은 기본 `--p-photo 1.0`입니다.
- 입력 파일, 선택한 정책·파라미터, 변환 결과는 출력 폴더의 `augmentation_manifest.jsonl`에 기록됩니다.

## 두 버전의 차이

| | `main` (CSV 버전) | `standalone-csv-free` (현재 브랜치) |
|---|---|---|
| 주 실행 파일 | `platform_aware_augment.py` | **`platform_aware_augment_standalone.py`** |
| 실행 시 원본 측정 CSV | 필요 | **불필요** |
| 변형 값 선택 | CSV의 개별 행 참조 | **코드 내장 개별 실측 조합 360개**에서 선택 |
| 용도 | 측정 근거를 직접 조회하며 실험 | 측정 정책을 고정해 학습 데이터 전처리 |

- [main 브랜치로 이동](https://github.com/kimmyju/Per_Platform_aug/tree/main)
- [Standalone v2 전체 프로토콜(한국어)](./README_standalone_KR.md)

## 측정 근거와 재현성에 관한 주의

기본 Standalone 정책은 **12개 프로파일, 65개 해상도·방향 조건, 360개 결합된 실측 변형값**을 내장합니다. Instagram의 중복 화질은 한 프로파일로 통합됩니다. 단, 실제 플랫폼 인코더를 완전히 복제하는 코드는 아닙니다. Instagram 중앙 크롭, 보간법, 인코더 프리셋, 플랫폼 25% 혼합 비율 등의 일부는 **구현상 가정·학습 설정값**입니다.

밝기·대비는 인코딩 뒤 측정해 목표값과 잔차 보정을 시도하지만, 코드의 광도 추정기는 원래 포렌식 측정기와 동일하지 않습니다. **최종 출력의 SSIM, PSNR, 압축률, 밝기·대비가 실측 분포를 재현하는지는 별도로 검증해야 합니다.** 측정 CSV 원본과 학습·평가 데이터는 이 공개 저장소에 포함하지 않았습니다.
