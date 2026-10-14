# XCA 영상의 관상동맥 협착 탐지: 전처리와 YOLO 성능 비교

관상동맥 X-ray Coronary Angiography(XCA) 영상에서 협착(stenosis)을 탐지하기 위해 독립적으로 진행한 실험을 정리했습니다. ARCADE 데이터의 병변 주변을 살펴보고, 전처리와 탐지 모델을 바꾸며 성능을 비교했습니다.

아래 결과는 당시 노트북에 저장된 출력에서 가져왔습니다. 이번 문서 정리에서 모델을 다시 학습하지는 않았습니다.

## 1. 프로젝트 배경

실험의 출발점은 **“협착 주변의 대비와 혈관 윤곽을 강화하면 탐지 성능도 좋아질까?”**라는 질문이었습니다. 영상 전체가 선명해지는 것과 작은 병변을 잘 찾는 것은 다를 수 있어, 병변 주변의 변화와 실제 탐지 결과를 함께 비교했습니다.

ARCADE stenosis의 COCO annotation을 분석하고 bbox 분포를 확인했습니다. 학습 데이터는 **1,000장·annotation 1,625개**, 검증 데이터는 **200장·annotation 406개**였습니다. 이후 협착만 탐지하도록 라벨을 변환했습니다.

데이터 버전·라이선스·환자별 분할 여부는 추가 확인이 필요합니다. 의료영상 원본과 병변 crop, 전처리 미리보기는 공개 권한이 확인되지 않아 로컬에 보존하고, 저장소에는 집계 그래프만 담았습니다.

## 2. 전처리 실험

같은 병변 주변을 잘라 CLAHE의 국소 대비, Gamma의 밝기 변화, Frangi의 혈관 강조 효과를 비교했습니다. Black-hat과 조합 전처리도 살펴봤고, Unsharp와 Top-hat은 시각화로 탐색했습니다.

CLAHE, Gamma, Frangi와 세 가지 조합, Black-hat의 **7종**을 배치 처리했습니다. 각 전처리의 학습 **1,000장**, 검증 **200장** 이미지가 생성된 기록을 확인했습니다. 이 결과는 `data/preprocessed/`에 이미지를 만든 단계이며, 이후 YOLO 학습 폴더로 복사하는 단계와는 구분했습니다.

육안 비교에 더해 영상 전체와 정답 병변 ROI의 대비, 경계 밀도, 선명도 지표를 계산했습니다. CLAHE와 CLAHE+Gamma는 원본보다 대비와 경계 밀도를 높였습니다. 다만 노이즈도 함께 증가할 수 있어, 영상 품질 지표만으로 결론을 내리지 않고 두 전처리를 YOLO 학습에서 비교했습니다.

전처리별 설정과 품질 지표는 [실험일지](docs/experiment_summary.md)에 정리했습니다.

## 3. YOLO 학습 결과

Original, CLAHE, CLAHE+Gamma를 사전학습 YOLO11n에서 각각 **100 epoch** 학습했습니다. 아래는 검증 **200장·406개 annotation**에 대한 최종 `best.pt` 검증 로그의 수치입니다.

| 입력 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.364 | 0.387 | 0.320 | 0.121 |
| CLAHE | 0.381 | 0.390 | 0.320 | 0.123 |
| CLAHE + Gamma | 0.381 | 0.387 | 0.330 | 0.127 |

CLAHE+Gamma의 mAP50-95는 원본 대비 **+0.006(0.6%p), 상대 +4.96%**였습니다. 이 조건에서는 CLAHE 단독보다 CLAHE+Gamma의 결과가 높았지만, 영상 대비가 크게 올라간 것에 비하면 탐지 지표의 차이는 작았습니다.

![YOLO11n의 원본·CLAHE·CLAHE+Gamma 비교](results/figures/05A_yolo11_result_aggregation__cell_006_output_01.png)

그래프와 표의 출처, 최종 검증과 최고 epoch 집계의 차이는 [결과 설명](results/README.md)에 정리했습니다.

## 4. 추가 모델 비교

같은 전처리가 다른 탐지기에서도 유효한지, 모델 크기를 늘리면 결과가 달라지는지 비교했습니다. 실제 실행한 추가 탐지기는 torchvision Faster R-CNN이며, YOLO11n과 YOLO11s도 별도로 비교했습니다.

| 실험 | 학습 기록 | 결과 |
|---|---|---|
| Faster R-CNN ResNet50-FPN, Original / CLAHE+Gamma | 각각 30 epoch | 최대 AP50 0.295686 / 0.290947 |
| Faster R-CNN ResNet50-FPN-v2, CLAHE+Gamma | 30 epoch | 최대 AP50 0.268508 |
| Faster R-CNN ResNet50-FPN, CLAHE+Gamma 단일 실행 | 100 epoch | 최대 AP50 0.310905 |
| Faster R-CNN MobileNetV3-FPN | 집계 결과만 남아 있음 | AP50 0.185888, 최고 checkpoint epoch 5. 전체 로그와 총 epoch는 확인 필요 |
| YOLO11n / YOLO11s, CLAHE+Gamma | 각각 50 epoch | 최종 mAP50-95 0.122 / 0.121 |

ResNet50-FPN의 30 epoch 비교에서는 원본이 CLAHE+Gamma보다 높았습니다. YOLO11s도 50 epoch 조건의 mAP50-95에서 YOLO11n을 넘지 못했습니다. 전처리나 모델 크기를 바꾸는 것이 모든 조건에서 개선으로 이어지지는 않았습니다.

Faster R-CNN과 Ultralytics는 평가 방식이 달라 하나의 종합 순위로 해석하지 않았습니다. 실행별 기록과 평가 기준은 [실험일지](docs/experiment_summary.md)에 남겼습니다.

커스텀 backbone 연결과 segmentation·ROI 확대도 후속 작업으로 다뤘습니다. Swin·ConvNeXt·ViT 이름의 스켈레톤과 모의 학습 기록은 있지만, 실제 backbone을 연결한 협착 탐지 결과는 없습니다. Segmentation 변환과 2단계 ROI 재탐지도 실제 ARCADE 성능 검증이 남아 있습니다.

## 5. 주요 트러블슈팅

- **YOLO 학습 폴더의 검증 이미지 누락:** 전처리 이미지는 생성되어 있었지만, 이후 YOLO 폴더로 옮긴 데이터의 개수 확인에서 검증 이미지 누락이 발견되었습니다. 학습에 사용할 CLAHE와 CLAHE+Gamma 이미지를 각각 200장 동기화하고 이미지·라벨 개수가 맞는지 확인한 뒤 학습했습니다. 다른 전처리까지 모두 복구했다는 기록은 없습니다.
- **추가 모델 실행 환경:** 원본 노트북에는 MMCV wheel을 구하지 못해 torchvision Faster R-CNN으로 진행했다는 설명이 남아 있습니다. 파일명에 있는 Swin·ConvNeXt를 실제 학습한 결과와는 구분했습니다.
- **노트북 실행 순서와 경로:** 확장 실험과 집계 과정에 선행 정의 셀 오류와 전처리 경로 불일치가 남아 있습니다. 해당 작업의 전체 재실행 성공 여부는 확인이 필요합니다.
- **결과 집계 기준과 폴더 재사용:** 최종 검증, 최고 mAP50 epoch, 최고 mAP50-95 epoch를 구분하고, 같은 폴더를 사용한 서로 다른 실행도 별도 기록으로 남겼습니다.

오류 메시지와 복구 과정은 [실험일지](docs/experiment_summary.md), 환경과 남은 수정 사항은 [재현성 문서](docs/reproducibility.md)에 정리했습니다.

## 6. 배운 점과 남은 과제

영상 대비를 높이는 것과 탐지 성능을 높이는 것은 따로 확인해야 한다는 점을 배웠습니다. CLAHE는 영상 품질 지표를 크게 높였지만, YOLO11n의 최종 mAP50-95에서는 CLAHE+Gamma가 더 높았습니다. 다른 탐지기에서는 원본이 더 좋은 결과도 확인했습니다. 전처리는 사용할 모델과 평가 조건까지 함께 비교해야 했습니다.

데이터를 생성하는 작업과 학습 입력을 준비하는 작업도 각각 확인해야 했습니다. 파일 개수와 라벨을 점검하고, 실행 조건과 결과 선택 기준을 함께 남기는 것이 이후 비교에 필요했습니다.

별도 test 평가, 환자 단위 분할과 데이터 누출 검사, 여러 seed의 반복 실험, 통계적 유의성과 신뢰구간, 외부 검증, 추론 지연시간은 추가 확인이 필요합니다. 현재 결과로 임상적 효용을 검증한 것은 아닙니다. ROI 확대 방식도 정답 기반 학습 crop과 예측 기반 추론 crop의 차이, 첫 단계에서 놓친 병변을 복구하지 못하는 한계를 평가해야 합니다.

단계별 진행 과정은 [실험일지](docs/experiment_summary.md), 다시 실행하기 위한 데이터 구조와 설치 절차는 [재현성 문서](docs/reproducibility.md)에 있습니다. 원본 보존과 공개 범위는 [공개 검토 문서](docs/publication_review.md)에 정리했습니다.
