# XCA 협착 탐지 실험일지

데이터를 살펴보는 작업부터 전처리, YOLO 학습, 추가 모델 비교까지 진행한 내용을 단계별로 정리했다. 노트북 11개와 Python 스크립트 3개에 남아 있는 수치와 오류를 기준으로 썼으며, 이번 문서 정리에서 다시 학습하지는 않았다. [README](../README.md)에서는 주요 흐름과 결과를 소개하고, 이 일지에는 설정·셀 위치·복구 과정까지 남겼다. 아래의 셀 번호는 원본에서 1부터 세는 위치다.

코드와 출력이 같은 시점에 저장된 것은 아닐 수 있다. 그래서 학습 결과가 있는 작업, 실행 중 오류가 난 작업, 코드만 있고 결과가 없는 작업을 나누어 적었다. 학습을 끝낸 것과 성능이 좋아진 것도 구분했다.

## 1. 데이터를 먼저 살펴보기

관련 노트북: `01_dataset.ipynb`

전처리에 앞서 협착 라벨이 어떻게 구성되어 있고 병변이 어느 정도 크기인지 살펴봤다. COCO JSON을 읽고 범주별 개수와 bbox 면적 분포를 확인한 다음, 이미지 5장에 bbox와 polygon을 겹쳐 표시했다.

cell 15/16에는 학습 이미지 **1000장**, annotation **1625개**가 나온다. cell 21의 집계는 category 26의 협착 annotation만 1625개다. cell 25/31에서 bbox 면적 평균은 **4836.93 px²**, 중앙값은 **2782.64 px²**였다. 평균 너비는 **63.7 px**, 높이는 **66.9 px**였다. 작은 병변 주변의 변화도 살펴볼 필요가 있어 이후 전처리 비교에서는 ROI를 함께 봤다.

cell 8의 총 **3000장**은 데이터 전체 검색 결과라 협착 학습 이미지 수와는 다르다. 이 데이터 분석에는 실행 결과가 남아 있다. 시각화한 의료영상 5장은 공개 권한을 확인하기 전까지 로컬에만 보존했다.

## 2. 병변 주변의 대비와 혈관 강조 방법 비교하기

관련 노트북: `02_preprocessing.ipynb`

같은 병변을 두고 전처리별로 무엇이 달라지는지 보기 위해 bbox 주변을 padding **40**으로 잘랐다. CLAHE의 clipLimit과 tileGrid를 바꾸는 비교부터 Gamma, Frangi, Black-hat, 조합 전처리까지 시각화했다. Unsharp와 Top-hat도 탐색했다.

cell 40의 최종 선택 값은 CLAHE clip **2.0**, tile **(4,4)**, Gamma **0.8**, Frangi sigmas **1..4**, Black-hat kernel **15**다. Frangi는 `gamma=15`, `black_ridges=True`로 적용했다. 함수 기본값과 중간 튜닝 값은 최종 선택 값과 다를 수 있어 순서대로 실행해야 한다.

이후 CLAHE, Gamma, Frangi, CLAHE+Gamma, CLAHE+Gamma+Frangi, Gamma+CLAHE+Frangi, Black-hat의 7종을 배치 처리했다. cell 44/45에는 각 전처리의 **train=1000, val=200**이 남아 있다. 저장 대상은 `data/preprocessed/<pipeline>/{train,val}/`이다. 이 기록은 **전처리 이미지 생성 단계**에 해당하며, YOLO 학습 폴더의 이미지·라벨 준비 여부를 검사한 결과는 아니다.

Unsharp와 Top-hat은 이 7종 배치 처리나 YOLO 학습 비교까지 진행한 항목은 아니다. 전처리와 ROI를 보여 주는 의료영상 포함 시각화 12개는 비공개로 보존했다.

## 3. 눈으로 본 변화를 숫자로 확인하기

관련 노트북: `03_quality_metrics.ipynb`

육안 비교에 더해 영상 전체와 정답 bbox ROI에서 entropy, 표준편차 대비, Canny 경계 밀도(50/150), Laplacian variance, Brenner focus를 계산했다. 한 이미지 안의 병변별 평균을 먼저 구한 뒤 이미지별 평균을 냈고, 정답 병변이 없는 이미지는 ROI 지표를 0으로 포함했다. cell 5에는 학습 이미지 **1000개**를 평가한 기록이 있고, cell 7에는 원본을 포함한 8개 전처리의 평균 표와 그래프가 있다.

CLAHE의 ROI contrast는 **34.135542**, 원본은 **17.958918**이었다. 대비 지표에서는 차이가 있었지만, 선명도나 경계 지표가 커진 이유에 노이즈도 포함될 수 있다. 그래서 영상 품질 지표와 실제 탐지 결과를 나누어 비교했다.

cell 7의 주요 평균 지표는 다음과 같다. 나머지 전처리를 포함한 전체 표는 [추출 CSV](../results/tables/03_quality_metrics__cell_007_output_03.csv)에 있다.

| 전처리 | ROI contrast | ROI edge density | global Laplacian variance |
|---|---:|---:|---:|
| Original | 17.958918 | 0.074510 | 230.966124 |
| CLAHE | 34.135542 | 0.240672 | 548.082601 |
| CLAHE+Gamma | 32.315488 | 0.224897 | 521.803766 |

품질 지표 계산과 그래프에는 실행 결과가 남아 있다. 다만 cell 9의 순위·종합 점수 계산은 미실행이므로, 그 계산까지 끝냈다고 정리하지 않았다.

## 4. YOLO 입력을 준비하면서 검증 이미지 누락 확인하기

관련 노트북: `03.5_Data Preparation.ipynb`

같은 라벨을 원본과 전처리 이미지에 사용할 수 있도록 COCO bbox를 YOLO 형식으로 정규화하고, category **26→0**으로 바꿨다. torchvision 쪽은 배경 class `0`, 협착 class `1`을 사용했다. 전처리 이미지 생성이 끝난 다음에는 별도로 `data/yolo/<pipeline>/images/{train,val}/`에 이미지를 복사하고 라벨을 준비해야 했다.

cell 6에서 원본 이미지는 **train=1000, val=200**이었다. cell 8의 라벨 변환 결과는 학습 bbox **1625개**, 빈 라벨 **3개**, 검증 bbox **406개**, 빈 라벨 **0개**다.

문제는 cell 10의 **YOLO 학습 폴더 개수 확인**에서 드러났다. 전처리 6종의 검증 이미지는 **0장**, 라벨은 **200개**여서 `CHECK`가 나왔다. 이는 `data/preprocessed/`의 이미지 생성이 아니라, 이후 `data/yolo/`에 학습 입력을 준비하는 단계의 누락 기록이다. 원본은 `OK`였지만 전처리 입력은 일부 준비가 빠진 상태였다. 초기 경로·복사 셀에는 실행 번호가 없고, 현재 코드에 있는 Black-hat도 당시 출력표에는 없어 처리 여부를 확인해야 한다.

이 누락은 이후 `04`에서 학습에 사용할 CLAHE와 CLAHE+Gamma 이미지를 다시 동기화하며 해결한 기록이 있다. 다른 전처리까지 모두 복구했다고 볼 기록은 없다.

단계별 경로와 기록을 나누면 다음과 같다. 두 노트북의 출력은 서로 다른 단계의 결과라, 전처리 생성 완료와 YOLO 입력 누락을 같은 작업의 성공·실패로 보지 않았다.

| 단계 | 대상 경로 | 남아 있는 기록 |
|---|---|---|
| 전처리 이미지 생성 (`02`) | `data/preprocessed/<pipeline>/{train,val}/` | 7종 각각 train=1000, val=200 |
| YOLO 이미지·라벨 준비 (`03.5`) | `data/yolo/<pipeline>/images/{train,val}/`, `labels/{train,val}/` | 전처리 6종의 val 이미지=0, 라벨=200으로 CHECK. 원본은 OK |
| 선택한 입력 동기화 (`04`) | YOLO의 CLAHE·CLAHE+Gamma 이미지 폴더 | 각각 train=1000, val=200 복사 후 Original 포함 세 입력 ready=True |

## 5. 같은 YOLO11n으로 전처리 효과 비교하기

관련 노트북: `04_train_yolo(cursor).ipynb.ipynb`, `05A_yolo11_result_aggregation.ipynb`

영상 품질 지표와 탐지 결과가 같은 방향으로 움직이는지 보기 위해 Original, CLAHE, CLAHE+Gamma를 사전학습 YOLO11n에서 비교했다. 학습 설정은 **100 epoch, imgsz640, batch16, seed42**였다.

`04` cell 4에서 CLAHE와 CLAHE+Gamma 이미지를 각각 **train1000, val200**으로 복사했다. cell 5/6에서 세 입력 모두 `ready=True`를 확인했고, cell 7에는 세 실험의 100 epoch 완료와 `best.pt` 최종 검증 로그가 남아 있다. 검증은 모두 **200장·406 instance**다.

| 원본·셀 | 실험 | 완료 epoch | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 04 cell 7 | `yolo11n_original` | 100 | 0.364 | 0.387 | 0.320 | 0.121 |
| 04 cell 7 | `yolo11n_clahe` | 100 | 0.381 | 0.390 | 0.320 | 0.123 |
| 04 cell 7 | `yolo11n_clahe_gamma` | 100 | 0.381 | 0.387 | 0.330 | 0.127 |

이 비교에서 CLAHE+Gamma의 mAP50-95는 원본 **0.121→0.127**, 차이는 **0.006**, 상대 변화는 **+4.96%**였다. 품질 지표의 변화 폭만큼 탐지 지표가 크게 달라지지는 않았지만, 이 조건에서는 CLAHE+Gamma가 가장 높았다.

### 결과표를 만들면서 구분한 두 가지 수치

`05A` cell 3은 CSV에서 최고 mAP50-95 epoch를 고른 표다. Original, CLAHE, CLAHE+Gamma 순서로 최고 epoch는 **74/86/90**, mAP50-95는 **0.12102/0.12320/0.12703**이었다.

이후 cell 4의 `use_manual=True`는 최종 검증 로그의 **0.121/0.123/0.127** 등을 수동으로 넣었다. cell 5/6의 차이 계산과 그래프는 이 수치를 사용한다. 수동 입력 값은 `04` 최종 검증 로그와 일치하지만, 앞선 CSV 최고 epoch 표와는 선택 기준이 달라 따로 보존했다. 실제 원본 CSV와 학습 가중치는 ZIP에 없어 새 환경에서의 재현은 확인이 필요하다.

## 6. Faster R-CNN과 YOLO 모델 크기 비교하기

관련 노트북: `05B_mmdet_swin_convnext.ipynb`

YOLO11n에서 전처리를 비교한 다음, 탐지기를 바꾸거나 모델 크기를 늘렸을 때도 같은 효과가 있는지 살펴봤다. 파일명에는 MMDetection, Swin, ConvNeXt가 들어 있지만 실제 구현과 실행 기록은 **torchvision Faster R-CNN**이다. 원본 노트북에는 Python 3.12 환경에서 MMCV wheel을 구하지 못해 이 방식으로 진행했다는 설명이 남아 있다. 평가는 직접 구현한 VOC 11-point AP50(score ≥ 0.05)을 사용했다.

### Faster R-CNN 결과

cell 7은 ResNet50-FPN+CLAHE+Gamma의 100 epoch 단일 실행, cell 8은 세 조합의 30 epoch 비교다. cell 7의 최대 AP50은 **0.3109045754**(소수 열 자리로 반올림)이고, 아래 표는 더 짧게 반올림했다. 더 정밀한 값과 epoch 기록은 `frcnn_history_summary.csv`와 개별 기록 CSV에 있다.

| 원본·셀 | 모델·입력 | 기록된 epoch | 최고 epoch | 최대 AP50 | 마지막 AP50 |
|---|---|---:|---:|---:|---:|
| 05B cell 7 | ResNet50-FPN / CG 단일 실행 | 100 | 4 | 0.310905 | 0.284617 |
| 05B cell 8 | ResNet50-FPN / Original 조합 비교 | 30 | 3 | 0.295686 | 0.255291 |
| 05B cell 8 | ResNet50-FPN / CG 조합 비교 | 30 | 4 | 0.290947 | 0.234812 |
| 05B cell 8 | ResNet50-FPN-v2 / CG 조합 비교 | 30 | 3 | 0.268508 | 0.216537 |
| 05B cell 9 | MobileNetV3-FPN / CG 집계만 | 확인 필요 | 5(checkpoint metadata) | 0.185888(출력 정밀도) | 확인 필요 |

여기서 CG는 CLAHE+Gamma다. 30 epoch 비교에서는 원본의 AP50이 CG보다 높았다. 학습을 끝낸 것과 전처리로 성능이 개선된 것은 다른 결과였다.

100 epoch 단일 실행과 30 epoch CG 비교는 같은 폴더명을 사용했다. 이후 `05B` cell 9와 `05E`의 집계는 CG **0.290947**을 보여 주므로, 앞선 **0.310905** 기록과 합치지 않았다. MobileNet은 집계 표와 최고 가중치 epoch 5만 남아 있어 총 학습 epoch와 전체 로그를 추가로 확인해야 한다.

학습 loss가 줄어드는 동안 검증 AP는 최고값보다 낮아지는 구간이 있다. 일반화 성능을 더 살펴볼 근거는 되지만, 이 기록만으로 과적합의 원인을 확정하기는 어렵다.

### YOLO11n과 YOLO11s의 50 epoch 비교

cell 11에서는 CLAHE+Gamma 입력으로 YOLO11n과 YOLO11s를 각각 50 epoch 학습했다. 두 실행 모두 완료 로그가 남아 있다.

| 원본·셀 | 실험 | 완료 epoch | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 05B cell 11 | `yolo11n__clahe_gamma` | 50 | 0.346 | 0.404 | 0.296 | 0.122 |
| 05B cell 11 | `yolo11s__clahe_gamma` | 50 | 0.356 | 0.401 | 0.310 | 0.121 |

YOLO11s의 mAP50은 높았지만 mAP50-95에서는 n을 넘지 못했다. 모델을 크게 만든 효과는 어떤 지표로 비교하는지도 함께 봐야 했다. 앞의 100 epoch 전처리 비교와는 학습 길이가 다르므로 한 조건처럼 묶지 않았다.

Faster R-CNN의 AP50과 Ultralytics mAP50도 평가 방식이 달라 직접적인 종합 순위로 해석하지 않았다.

## 7. 커스텀 backbone 연결을 위한 모의 실험

관련 노트북: `05C_yolo11_custom_backbone_skeleton.ipynb`

외부 backbone을 연결하기 위한 인터페이스를 살펴보면서 ConvNeXt·Swin·ViT 이름의 합성곱 스켈레톤과 TinyFPN을 만들었다. cell 3에는 특징 맵 크기, cell 6에는 Swin 이름의 모의 탐지기 출력이 남아 있다. 실제 Swin transformer나 정식 ConvNeXt·ViT 탐지 모델을 연결한 것은 아니다.

cell 9는 랜덤 이미지와 랜덤 타깃으로 MSE 회귀를 실행한 작업이다. loss는 **4.5122→2.0322→1.2110**으로 줄었지만, 이 값은 협착 탐지 성능을 의미하지 않는다. 인터페이스를 시험하는 모의 학습 결과로 남겼다.

같은 노트북의 cell 8에는 별도로 기존 YOLO11n을 50 epoch 학습한 기준 결과가 있다. 이 결과를 커스텀 backbone 학습 결과로 쓰지 않았다.

| 원본·셀 | 실험 | 완료 epoch | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 05C cell 8 | `yolo11n_Ctrack_clahe_gamma` | 50 | 0.346 | 0.404 | 0.296 | 0.122 |

`05B`의 n 수치와 같지만 독립적인 반복 실험의 근거로 보기는 어렵다. seed·설정·원본 산출물 확인이 필요하다. 정식 Swin·ConvNeXt·ViT·InternImage 탐지 모델의 통합과 학습은 미완료다.

## 8. 확장 비교와 결과 집계에서 남은 오류

### 본실험 조합을 확장하려던 작업

관련 노트북: `05D_backbone_experiment_matrix.ipynb`

ResNet과 MobileNet 비교를 확장할 수 있도록 AdamW와 CosineAnnealingLR을 사용하는 학습 코드를 구성했다. YOLO와 ConvNeXt 선택 코드도 있지만, 본실험 결과는 남아 있지 않다.

cell 4에는 **`NameError: name 'random' is not defined`**가 있다. 선행 import 정의 셀은 실행하지 않은 상태이며, 다른 코드 셀에는 출력이 없다. 따라서 이 노트북의 본실험은 미완료로 남겼다.

경로도 점검이 필요하다. CLAHE+Gamma를 `02_clahe_gamma`로 참조하지만 실제 전처리 폴더는 `04_clahe_gamma`다. 정답 병변이 없는 이미지 선택 방식도 `05B`와 다르다. 이 부분을 해결하지 않은 상태에서 같은 조건의 비교라고 볼 수는 없다.

### 여러 모델의 결과를 한 번에 모으려던 작업

관련 노트북: `05E_compare_all_results.ipynb`

Faster R-CNN의 AP50과 YOLO의 최고 mAP50 epoch를 모아 모델별 표와 그래프를 만들었다. 다만 cell 4에는 **`NameError: name 'load_frcnn_rows' is not defined`**가 있고, cell 5~8에는 후속 표와 그래프가 함께 남아 있다. 처음부터 전체를 실행해 성공했는지는 확인이 필요하다.

후속 집계의 YOLO 최고 mAP50은 **0.345780**, 같은 epoch의 mAP50-95는 **0.12645**였다. `05A`는 최고 mAP50-95 epoch를 고르므로 이 둘을 그 표의 수치와 섞지 않았다. 오류와 후속 출력 모두 보존했고, 새로운 실행이 성공한 결과로 표현하지 않았다.

## 9. Segmentation 라벨과 ROI 확대를 위한 후속 구현

작은 병변을 확대하는 방식은 별도 데이터 준비와 추론 과정이 필요하다. 아래 작업에는 코드가 있지만 실제 ARCADE 학습·평가 결과는 없다.

### COCO polygon을 YOLO segmentation 라벨로 변환하기

관련 자료: `03.6_coco_to_yolo_seg.ipynb`, `06A_convert_coco_to_yolo_seg.py`

`03.6`은 변환기를 생성하거나 불러와 polygon을 정규화하고 split별 라벨을 저장하는 구성이다. 모든 코드 셀의 `execution_count=null`이고 출력이 없어 실행 여부를 확인할 수 없다. 본문에 적힌 학습·검증 데이터 기대치는 실제 segmentation 변환 결과가 아니다. 실제 mask 검수·학습·평가는 미완료다.

독립 스크립트 `06A`는 category **26→0** 변환, crowd·RLE·잘못된 polygon 제외, 좌표 정규화와 전처리별 라벨 복사를 구현한다. 정리 과정에서 합성 COCO 데이터로 기능을 검증했지만, 실제 ARCADE 변환 결과가 추가된 것은 아니다.

### 정답 병변 주변을 잘라 학습 입력 만들기

관련 스크립트: `06B_build_roi_crop_pipeline.py`

정답 polygon의 bbox를 가까운 병변끼리 묶고, 군집 크기를 제한한 뒤 여백을 포함해 자르는 코드다. 이미지를 **640** 크기로 조정하고 polygon 좌표를 crop 기준으로 다시 정규화한다.

실제 실행 로그·생성 이미지·탐지 성능 수치는 없다. 배경 이미지를 건너뛰고 정답을 사용해 자르므로, 이 입력의 평가를 전체 영상 탐지 결과와 바로 비교할 수는 없다.

### 추론에서도 예측 영역을 확대해 다시 탐지하기

관련 스크립트: `06C_roi_crop_cascade.py`

첫 모델의 예측 bbox를 군집으로 묶어 자르고, 두 번째 모델로 다시 탐지한 뒤 좌표를 원본 영상으로 돌려 NMS로 합치는 2단계 구조다. TP·FP·FN 계산 함수도 포함되어 있다.

두 모델의 실행 로그나 평가 수치는 없어 실제 실험은 확인이 필요하다. 첫 단계에서 아무 병변도 찾지 못하면 종료하므로, 놓친 병변을 두 번째 단계에서 복구할 수 없는 한계도 있다. 정답으로 자르는 학습 입력과 예측으로 자르는 추론 입력의 차이를 함께 평가해야 한다.

## 10. 이번 기록에서 정리한 점과 다음 확인

YOLO11n의 100 epoch 비교에서는 CLAHE+Gamma의 최종 mAP50-95가 원본보다 **0.006** 높았다. 하지만 영상 대비가 높아진 만큼 탐지 성능도 높아진 것은 아니었다. Faster R-CNN의 30 epoch 비교는 원본이 더 높았고, YOLO11s도 50 epoch 조건에서 n의 mAP50-95를 넘지 못했다.

검증 이미지 누락은 `04`에서 선택한 두 전처리를 동기화하며 해결했다. 반면 `05D`·`05E`의 실행 오류와 경로 문제, 커스텀 backbone 통합, segmentation과 ROI 확대 성능 검증은 남은 작업이다. 결과가 있는 비교와 구현만 진행한 작업을 함께 적어두는 것이 이 실험의 현재 범위를 설명하는 데 필요했다.

데이터 공개 권한, 정확한 실행 환경, 원본 CSV·학습 가중치, 환자 단위 분할, 별도 test 평가, 여러 seed의 반복 비교와 공통 평가 방식은 추가 확인이 필요하다. 관련 기술적인 점검 내용은 [재현성 문서](reproducibility.md), 수치별 출처는 [결과 설명](../results/README.md)에 남겼다.
