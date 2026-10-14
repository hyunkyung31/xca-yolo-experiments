# XCA 영상의 관상동맥 협착 탐지: 전처리와 YOLO 성능 비교

관상동맥 X-ray Coronary Angiography(XCA) 영상에서 협착(stenosis)을 탐지하기 위해 독립적으로 진행한 실험 기록이다. ARCADE 데이터를 살펴보고, 전처리로 병변 주변의 대비를 바꾼 뒤 YOLO11n의 탐지 결과를 비교했다. 이후 Faster R-CNN과 YOLO11s로 비교 범위를 넓혔다.

수치는 당시 노트북에 저장된 출력에서 가져왔다. 이번 정리에서 다시 학습한 결과는 아니며, 원본 데이터·학습 가중치·원본 `results.csv`는 제공된 ZIP에 없었다. 각 수치의 출처와 실행 상태는 [실험일지](docs/experiment_summary.md)에 함께 정리했다.

## 1. 실험을 시작한 배경

실험에서 다룬 질문은 **“협착 주변의 대비와 혈관 윤곽을 강화하면 탐지 성능도 좋아질까?”**였다. 영상 전체가 선명해지는 것과 작은 병변을 잘 찾는 것은 다를 수 있어, 전처리 결과를 병변 주변에서 살펴보고 실제 탐지 성능과 비교했다. 이 주제를 선택한 개인적인 계기는 기록에 남아 있지 않아 확인이 필요하다.

먼저 `01_dataset.ipynb`에서 COCO 형식의 annotation을 읽고 데이터 규모와 bbox 분포를 확인했다. 사용한 데이터는 **ARCADE stenosis**이며, `category_id=26`을 YOLO class `0`으로 바꿨다. torchvision에서는 배경이 `0`, 협착이 `1`이다.

- 학습 데이터: **1,000장, annotation 1,625개**. 협착 라벨이 없는 이미지는 3장이다.
- 검증 데이터: **200장, annotation 406개**. 협착 라벨이 없는 이미지는 0장이다.
- 학습 bbox 면적: 평균 **4,836.93 px²**, 중앙값 **2,782.64 px²**. 평균 너비는 63.7 px, 높이는 66.9 px이다.

초기 탐색에서 나온 총 3,000장은 데이터 ZIP 전체를 검색한 수다. 이번 협착 탐지의 학습 이미지 수와는 다르다. bbox 분포를 살펴본 뒤, 영상 전체뿐 아니라 작은 병변 주변을 따로 비교하는 방향으로 전처리 실험을 진행했다.

데이터 버전·공식 다운로드 주소·라이선스와 재배포 조건·환자별 분할 여부는 확인이 필요하다. 의료영상 원본과 병변 crop, 전처리 미리보기는 공개 권한이 확인되지 않아 로컬에 보존하고, 저장소에는 집계 그래프만 담았다.

## 2. 전처리 실험: 병변 주변에서 무엇이 달라지는지 보기

`02_preprocessing.ipynb`에서는 같은 병변 주변을 잘라 전처리 결과를 비교했다. CLAHE의 국소 대비, Gamma의 밝기 변화, Frangi의 혈관 강조 효과를 살펴보고, Black-hat과 여러 조합도 함께 적용했다. Unsharp와 Top-hat은 함수와 시각화로 탐색했다.

이후 이미지 전체에 적용해 저장한 전처리는 다음 7종이다.

- CLAHE, Gamma, Frangi
- CLAHE+Gamma, CLAHE+Gamma+Frangi, Gamma+CLAHE+Frangi
- Black-hat

7종 모두 학습 1,000장과 검증 200장을 처리한 출력이 남아 있다. Unsharp와 Top-hat은 이 배치 처리나 YOLO 성능 비교에 포함된 전처리는 아니다.

CLAHE는 `clipLimit`과 `tileGrid`를 바꿔가며 비교했다. 최종 선택 셀의 값은 `clipLimit=2.0`, `tileGrid=(4,4)`이다. Gamma는 `0.8`, Frangi는 `sigmas=1..4`, `gamma=15`, `black_ridges=True`, Black-hat kernel은 `15`로 기록되어 있다. 함수 기본값과 중간 튜닝 값이 다르므로 다시 실행할 때는 최종 선택 셀까지 순서대로 실행해야 한다.

육안 비교에 더해 `03_quality_metrics.ipynb`에서 이미지 품질 지표를 계산했다. 학습 영상의 entropy, 표준편차 대비, Canny edge density(50/150), Laplacian variance, Brenner focus를 전체 이미지와 정답 bbox ROI에서 구했다. 한 이미지에 병변이 여러 개면 병변별 평균을 먼저 계산하고, 그다음 이미지별 평균을 냈다. 정답 병변이 없는 이미지는 ROI 지표를 0으로 포함했다.

| 전처리 | ROI contrast | ROI edge density | global Laplacian variance |
|---|---:|---:|---:|
| Original | 17.958918 | 0.074510 | 230.966124 |
| CLAHE | 34.135542 | 0.240672 | 548.082601 |
| CLAHE+Gamma | 32.315488 | 0.224897 | 521.803766 |

`03_quality_metrics` cell 7의 평균 표에서는 CLAHE와 CLAHE+Gamma가 원본보다 대비와 경계 밀도를 높였다. 다만 이런 변화에는 노이즈 증가도 포함될 수 있다. 영상 지표만으로 탐지가 좋아졌다고 결론 내리지 않고, 이 두 전처리를 YOLO 학습에서 원본과 비교했다. 나머지 전처리 지표도 [추출 CSV](results/tables/03_quality_metrics__cell_007_output_03.csv)에 남겼다.

## 3. YOLO 학습: 전처리 효과를 탐지 결과로 비교하기

`03.5_Data Preparation.ipynb`에서 COCO bbox를 YOLO 라벨로 변환한 뒤, `04_train_yolo(cursor).ipynb.ipynb`에서 Original, CLAHE, CLAHE+Gamma를 비교했다. 세 실험 모두 사전학습 YOLO11n으로 100 epoch를 마친 기록이 있다. 검증 데이터는 200장, 협착 annotation은 406개다.

아래는 **최종 `best.pt` 검증 로그**의 수치다. 소수 정밀도는 당시 로그에 출력된 그대로다.

| 입력 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.364 | 0.387 | 0.320 | 0.121 |
| CLAHE | 0.381 | 0.390 | 0.320 | 0.123 |
| CLAHE + Gamma | 0.381 | 0.387 | 0.330 | 0.127 |

CLAHE+Gamma의 mAP50-95는 원본 대비 **+0.006(0.6%p), 상대 +4.96%**였다. 이 비교에서는 CLAHE 단독보다 CLAHE+Gamma의 결과가 높았다. 전처리에서 대비가 크게 올라간 것에 비하면 탐지 지표의 차이는 작았다.

![YOLO11n의 원본·CLAHE·CLAHE+Gamma 비교](results/figures/05A_yolo11_result_aggregation__cell_006_output_01.png)

검증 로그는 `04` cell 7, 그래프는 `05A` cell 6에 있다. `05A` cell 4는 최종 검증 수치를 수동 입력해 그렸으며, 입력 값이 앞선 로그와 일치한다. 같은 노트북의 cell 3에는 CSV에서 최고 mAP50-95 epoch를 고른 별도 표도 있어 두 결과를 나누어 보존했다.

## 4. 추가 모델 비교: 같은 전처리가 다른 조건에서도 유효할까?

YOLO11n에서 얻은 결과를 바탕으로 탐지기와 모델 크기를 바꿔 비교했다. `05B_mmdet_swin_convnext.ipynb`라는 파일명과 달리, 실제 실행한 탐지기는 torchvision Faster R-CNN이다. 원본 노트북에는 Python 3.12 환경에서 MMCV wheel을 구하지 못해 이 방식으로 진행했다는 설명이 남아 있다.

| 실험 | 학습 기록 | 결과 |
|---|---|---|
| Faster R-CNN ResNet50-FPN, Original / CLAHE+Gamma | 각각 30 epoch | 최대 AP50 0.295686 / 0.290947 |
| Faster R-CNN ResNet50-FPN-v2, CLAHE+Gamma | 30 epoch | 최대 AP50 0.268508 |
| Faster R-CNN ResNet50-FPN, CLAHE+Gamma 단일 실행 | 100 epoch | 최대 AP50 0.310905 |
| Faster R-CNN MobileNetV3-FPN | 집계 결과만 남아 있음 | AP50 0.185888, 최고 checkpoint epoch 5. 전체 로그와 총 epoch는 확인 필요 |
| YOLO11n / YOLO11s, CLAHE+Gamma | 각각 50 epoch | 최종 mAP50-95 0.122 / 0.121 |

ResNet50-FPN의 30 epoch 비교에서는 원본이 CLAHE+Gamma보다 높았다. YOLO11s도 50 epoch 조건의 mAP50-95에서 YOLO11n을 넘지 못했다. 전처리나 모델 크기를 바꾸는 것이 모든 조건에서 같은 방향의 개선으로 이어지지는 않았다.

Faster R-CNN은 직접 구현한 **VOC 11-point AP50(score ≥ 0.05)**로 평가했다. Ultralytics의 mAP50과 계산 방식이 달라 두 탐지기의 수치를 하나의 종합 순위로 해석하기는 어렵다. 또한 ResNet50-FPN의 CLAHE+Gamma 단일 실행과 이후 30 epoch 실행은 같은 결과 폴더를 사용했으므로 별도 기록으로 구분했다.

커스텀 backbone과 ROI 확대도 후속 작업으로 다뤘다. `05C`에는 Swin·ConvNeXt·ViT 이름의 스켈레톤과 Swin 이름의 모의 회귀 실행이 있지만, 실제 backbone을 연결해 협착 탐지를 학습한 결과는 없다. `03.6`, `06A`–`06C`의 segmentation 변환, ROI crop, 2단계 재탐지 역시 구현 단계이며 실제 ARCADE 성능 수치는 확인이 필요하다.

## 5. 진행 중 만난 문제와 해결하지 못한 부분

### 전처리 검증 이미지가 빠진 문제

`03.5`의 데이터 개수 확인에서 전처리 검증 이미지가 0장인데 라벨은 200개인 `CHECK` 출력이 나왔다. 원본 이미지와 라벨은 정상적으로 준비되어 있었다. 이후 `04`에서 CLAHE와 CLAHE+Gamma 검증 이미지를 각각 200장 복사했고, 선택한 세 입력 모두 준비 완료를 확인한 뒤 학습했다. 이 복구 기록은 학습에 사용한 두 전처리에 해당한다.

### 노트북 셀 실행 순서와 경로 문제

`05D`에는 `NameError: name 'random' is not defined`, `05E`에는 `NameError: name 'load_frcnn_rows' is not defined`가 남아 있다. 선행 정의 셀 실행 순서를 확인해야 하는 오류다. `05E`에는 이후 표와 그래프도 있지만, 전체를 처음부터 다시 실행해 성공했는지는 확인이 필요하다.

`05D`는 CLAHE+Gamma 경로를 `02_clahe_gamma`로 참조한다. 실제 배치 결과는 `04_clahe_gamma`에 있어 경로가 맞지 않는다. 이 부분은 수정하지 않은 후속 과제로 남겼다.

### 같은 실험에서도 집계 기준에 따라 수치가 달라진 문제

`05A`는 최고 mAP50-95 epoch, `05E`는 최고 mAP50 epoch를 고른다. 최종 YOLO 로그는 선택된 가중치를 다시 검증한 결과다. 이 세 값을 한 결과처럼 섞으면 비교가 달라져, [수치 출처](results/README.md)에 따로 설명했다. 고정된 결과 폴더를 재사용한 기록도 있어, 재실행할 때는 새 실험 이름을 쓰는 편이 좋다.

## 6. 결과를 돌아보며 정리한 점

이번 비교에서 확인한 것은 영상의 대비를 높이는 것과 탐지 성능을 높이는 것이 같은 문제는 아니라는 점이다. CLAHE는 품질 지표를 크게 높였지만 YOLO11n의 최종 mAP50-95에서는 CLAHE+Gamma가 더 높았다. 다른 탐지기에서는 원본이 더 좋은 결과도 나왔다. 전처리 효과는 사용할 모델과 평가 조건까지 함께 비교해야 했다.

실험을 돌리는 일만큼 결과를 남기는 방식도 비교에 영향을 줬다. 원본 검증 로그, 최고 epoch 표, 수동 입력한 비교표의 기준을 구분하고, 실패한 실행과 미완료 구현도 실험일지에 같이 남겼다.

이번 수치만으로 일반화 성능을 확정할 수는 없다. 별도 test 평가, 환자 단위 분할과 데이터 누출 검사, 여러 seed의 반복 실험, 통계적 유의성과 신뢰구간, 외부 검증, 추론 지연시간은 확인이 필요하다. 임상적 효용을 검증한 결과도 아니다. ROI 확대 방식은 정답으로 자른 학습 이미지와 예측으로 자른 추론 이미지의 차이를 평가해야 하고, 첫 단계에서 놓친 병변은 다음 단계로 넘어가지 못하는 한계가 있다.

## 실험 자료와 다시 실행하는 방법

```text
notebooks/                 # 노트북 11개
scripts/                   # 원본 06A/06B/06C와 보존 검증 도구
results/
  figures/                 # 원본 출력의 집계 그래프 5개
  tables/                  # 표 HTML/CSV, FRCNN epoch 기록, YOLO 최종 검증
  evidence/                # 지표·학습 완료·오류 로그
  image_manifest.json      # 이미지 22개의 출처·해시·공개 여부
docs/
  experiment_summary.md    # 단계별 실험일지
  reproducibility.md       # 환경과 재실행 절차
  publication_review.md    # 공개 범위와 원본 보존
  change_report.md         # 정리 작업의 변경·검증 내역
  source_manifest.json     # 원본 ZIP·14개 파일과 정리본 해시
```

원본 ZIP, 전체 출력과 의료영상은 저장소 밖에 보존했다. 동일 해시의 중복 파일은 없었고, 노트북의 반복 코드와 출력 출처도 유지했다.

기존 결과를 읽는 데는 데이터가 필요하지 않다. 다시 실행하려면 데이터와 가중치를 별도로 준비해야 한다. 데이터는 `data/arcade/arcade/stenosis/{train,val}/images/`와 `annotations/{train,val}.json`에 두고, 노트북은 `notebooks/`에서 실행한다. 데이터·코드·모델의 라이선스와 재배포 조건도 먼저 확인해야 한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_portfolio.py
cd notebooks
jupyter lab
```

당시 로그의 Python은 3.12.6이다. 전체 패키지 lockfile은 없어 미확인 버전을 임의로 지정하지 않았다. PyTorch/torchvision은 장비의 CPU·CUDA에 맞는 공식 배포 조합을 확인해야 하며, 과거 로그의 버전을 지금 설치할 수 있는지도 확인이 필요하다. 위 절차는 재실행을 위한 안내이고, 현재 환경에서 의료 데이터로 재현한 절차는 아니다. 자세한 환경 근거와 남은 문제는 [재현성 문서](docs/reproducibility.md)에 있다.

1. `01_dataset` → `02_preprocessing` → `03_quality_metrics` 순서로 데이터와 전처리를 살펴본다. `01`의 `--force-reinstall opencv-python` 셀은 환경을 바꾸므로 필요할 때만 실행한다.
2. `03.5_Data Preparation`에서 이미지·라벨을 준비하고 학습 대상의 train/val 개수가 일치하는지 확인한다. `04`에 CLAHE·CLAHE+Gamma의 검증 이미지 동기화 셀이 있다.
3. `04`의 `DEVICE=0`, batch, workers를 장비에 맞게 검토한 뒤 학습한다. 새 결과를 집계할 때는 `05A`의 `use_manual=True`를 `False`로 바꿔 과거 수치가 새 결과를 덮어쓰지 않게 한다.
4. `05B`는 `RUN_MATRIX=True`, `RUN_YOLO_SCALES=True`라서 전체 실행 시 추가 학습도 시작된다. 실행할 비교와 새 실험 이름을 먼저 정한다.
5. `05D`, `05E`는 오류와 경로 차이를 먼저 점검한다. `03.6`, `06A`–`06C`는 미완료 확장 작업으로 구분한다.

Segmentation 변환은 `python scripts/06A_convert_coco_to_yolo_seg.py --help`로 옵션을 보고 `--data-dir`를 지정한다. `--dry-run`에서도 빈 디렉터리가 생길 수 있다. `06B`는 명령줄에서 실행할 수 있고, `06C`는 두 모델과 crop 함수들을 받아 사용하는 모듈이다. Detection과 segmentation 라벨은 같은 디렉터리에 섞지 않는다.
