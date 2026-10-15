# XCA 영상의 관상동맥 협착 탐지: 전처리와 YOLO 성능 비교

관상동맥조영술(X-ray Coronary Angiography, XCA) 영상에서 협착 부위를 탐지하기 위해 이미지 전처리와 객체 탐지 모델을 비교한 프로젝트입니다.

ARCADE 데이터셋을 활용해 CLAHE, Gamma Correction 등 여러 전처리 방법을 적용하고, YOLO11과 Faster R-CNN 모델의 탐지 성능을 비교했습니다. 특히 작은 협착 부위의 탐지가 어려운 문제를 중심으로 전처리 효과와 모델별 차이를 살펴봤습니다.

## 데이터 출처

이 프로젝트는 [ARCADE 데이터셋](https://arcade.grand-challenge.org/)을 사용했습니다.

데이터는 [Zenodo 배포 페이지](https://zenodo.org/records/7981245)에서 안내하고 있으며, 공식 이용 조건은 CC BY-NC-SA입니다. 데이터 접근에는 별도의 [등록 절차](https://arcade.grand-challenge.org/registration/)가 있습니다.

이 저장소에는 실험 코드와 집계 결과를 정리했으며, 원본 의료영상과 annotation 데이터셋, 모델 가중치는 포함하지 않았습니다.

## 1. 프로젝트 배경

관상동맥조영술 영상의 협착 부위는 크기가 작거나 주변 혈관과의 명암 차이가 크지 않아 탐지가 어려운 경우가 있습니다.

초기 YOLO 모델을 학습했을 때도 mAP가 낮았고, 작은 협착 부위를 제대로 탐지하지 못하는 문제가 있었습니다.

이를 개선하기 위해 영상의 대비를 높이고 혈관 경계를 강조하는 전처리를 먼저 실험했습니다. 이후 실제 객체 탐지 모델에 적용해 영상의 시각적 품질 변화가 탐지 성능 개선으로 이어지는지 비교했습니다.

CLAHE, Gamma, Frangi 등 여러 전처리 기법을 적용했으며, YOLO11n뿐 아니라 Faster R-CNN과 YOLO11s까지 비교 범위를 확장했습니다.

![학습 데이터의 협착 bbox 면적 분포](results/figures/01_dataset__cell_023_output_01.png)

학습 데이터의 협착 영역은 작은 크기에 집중된 분포를 보였습니다. 이는 작은 병변을 고려한 전처리 실험을 진행한 배경 중 하나입니다.

## 2. 이미지 전처리 실험

협착 부위와 주변 혈관의 특징을 강조하기 위해 다양한 전처리 기법을 비교했습니다.

- **CLAHE:** 국소 영역의 대비 개선
- **Gamma Correction:** 영상 밝기 조절
- **Frangi Filter:** 혈관과 같은 선형 구조 강조
- **Black-hat:** 어두운 구조 강조
- **조합 전처리:** 여러 기법을 결합했을 때의 변화 비교

Unsharp Mask와 Top-hat도 추가로 시각화해 봤습니다.

CLAHE, Gamma, Frangi와 조합 전처리 등을 포함한 총 7종의 전처리를 배치 처리했으며, 각 전처리에서 학습 이미지 1,000장과 검증 이미지 200장을 생성했습니다.

### 전처리 품질 비교

육안으로 영상을 비교하는 것 외에도 병변 ROI(Region of Interest)를 기준으로 대비, 경계 밀도, 선명도 등을 계산했습니다.

![Original·CLAHE·CLAHE+Gamma의 병변 ROI 품질 지표](results/figures/preprocessing_roi_quality.png)

CLAHE와 CLAHE+Gamma는 원본 영상보다 대비와 경계 밀도가 높았습니다.

하지만 대비가 증가하면서 노이즈도 함께 강조될 수 있고, 영상 품질 지표가 높다고 실제 탐지 성능까지 좋아지는 것은 아닙니다.

따라서 Original, CLAHE, CLAHE+Gamma를 동일한 YOLO11n 학습 조건에서 비교했습니다.

전처리별 설정과 세부 실험 내용은 [실험일지](docs/experiment_summary.md)에 정리했습니다.

## 3. YOLO11n 학습 및 성능 비교

Original, CLAHE, CLAHE+Gamma 데이터를 사용해 사전학습된 YOLO11n을 각각 100 epoch 학습했습니다.

검증 데이터는 이미지 200장, annotation 406개이며, 아래 표는 각 실험의 최종 `best.pt` 검증 결과입니다.

| 입력 데이터 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.364 | 0.387 | 0.320 | 0.121 |
| CLAHE | 0.381 | 0.390 | 0.320 | 0.123 |
| CLAHE + Gamma | 0.381 | 0.387 | **0.330** | **0.127** |

### 결과 분석

CLAHE+Gamma를 적용한 모델의 mAP50-95는 0.127로, Original의 0.121보다 0.006(0.6%p) 높았습니다.

상대적으로는 약 4.96% 개선된 수치입니다.

![YOLO11n 최종 best.pt 검증의 mAP50·mAP50-95 비교](results/figures/yolo11n_final_validation.png)

세 가지 입력 중 CLAHE+Gamma가 가장 높은 성능을 보였지만, 개선 폭이 크지는 않았습니다.

전처리로 영상의 대비와 경계가 뚜렷해졌더라도 모델의 탐지 정확도가 그만큼 높아지는 것은 아니라는 점을 확인했습니다.

또한 이 결과만으로 작은 협착 병변의 탐지 성능이 별도로 개선되었다고 판단할 수는 없습니다.

평가 수치와 그래프의 세부 기준은 [결과 설명](results/README.md)에 정리했습니다.

## 4. 추가 객체 탐지 모델 비교

전처리 효과가 다른 모델에서도 비슷하게 나타나는지 알아보기 위해 Faster R-CNN을 추가로 실험했습니다.

또한 YOLO11n보다 큰 YOLO11s를 학습해 모델 크기에 따른 성능 차이도 비교했습니다.

| 실험 모델 | 학습 조건 | 결과 |
|---|---|---|
| Faster R-CNN ResNet50-FPN, Original | 30 epoch | 최대 AP50 0.295686 |
| Faster R-CNN ResNet50-FPN, CLAHE+Gamma | 30 epoch | 최대 AP50 0.290947 |
| Faster R-CNN ResNet50-FPN-v2, CLAHE+Gamma | 30 epoch | 최대 AP50 0.268508 |
| Faster R-CNN ResNet50-FPN, CLAHE+Gamma | 별도 100 epoch 실험 | 최대 AP50 0.310905 |
| Faster R-CNN MobileNetV3-FPN | 제한된 집계 결과 | AP50 0.185888 |
| YOLO11n, CLAHE+Gamma | 50 epoch | mAP50-95 0.122 |
| YOLO11s, CLAHE+Gamma | 50 epoch | mAP50-95 0.121 |

### Faster R-CNN 결과

ResNet50-FPN의 30 epoch 실험에서는 Original이 CLAHE+Gamma보다 높은 AP50를 보였습니다.

YOLO11n에서 CLAHE+Gamma가 더 좋은 결과를 보인 것과는 다른 양상이었습니다.

![ResNet50-FPN의 30 epoch 비교와 별도 100 epoch 실행의 VOC AP50 변화](results/figures/frcnn_ap50_history.png)

30 epoch 비교와 100 epoch 단일 실험은 서로 다른 실행 조건이므로 구분해서 해석했습니다.

또한 Faster R-CNN의 VOC 11-point AP50와 YOLO의 mAP50는 평가 방식이 달라 수치를 직접 비교해 모델의 우열을 판단하지 않았습니다.

### YOLO11n과 YOLO11s 비교

CLAHE+Gamma 데이터로 각각 50 epoch 학습했을 때 YOLO11n의 mAP50-95는 0.122, YOLO11s는 0.121이었습니다.

이번 실험에서는 모델 크기를 늘리는 것만으로 성능이 개선되지는 않았습니다.

추가로 커스텀 Backbone, Segmentation, ROI 기반 재탐지 방식도 탐색했지만, 실제 협착 탐지 성능을 검증하는 단계까지 이어지지는 못했습니다.

모델별 세부 조건은 [실험일지](docs/experiment_summary.md)에서 확인할 수 있습니다.

## 5. 주요 문제와 해결 과정

### 5.1 전처리 데이터의 검증 이미지 누락

전처리 이미지를 생성한 뒤 YOLO 학습용 폴더를 구성하는 과정에서 검증 이미지가 일부 누락된 문제가 있었습니다.

전처리 결과물이 생성되었다고 해서 학습에 사용할 데이터 구성까지 완료된 것은 아니었습니다.

CLAHE와 CLAHE+Gamma의 검증 이미지를 각각 200장으로 맞추고, 이미지와 라벨 개수를 확인한 뒤 학습을 진행했습니다.

이 과정을 통해 데이터 전처리뿐 아니라 학습 데이터셋을 구성한 이후의 무결성 검사도 중요하다는 점을 알게 되었습니다.

### 5.2 Faster R-CNN 실행 환경 문제

추가 모델 실험 과정에서는 MMCV 의존성에 맞는 wheel을 확보하기 어려운 문제가 있었습니다.

이에 torchvision 기반 Faster R-CNN으로 실험을 진행했습니다.

Swin, ConvNeXt, ViT 기반 커스텀 Backbone도 검토했지만, 실제 Backbone을 연결한 협착 탐지 성능까지 평가하지는 못했습니다.

### 5.3 실험 환경과 결과 관리

여러 전처리와 모델을 비교하면서 노트북 실행 순서, 입력 경로, 결과 저장 폴더를 관리하는 데 어려움이 있었습니다.

특히 서로 다른 실행에서 동일한 출력 폴더를 사용하면 최종 검증 결과와 특정 epoch의 최고 성능을 혼동하기 쉬웠습니다.

이후 결과를 비교할 때 학습 조건, 체크포인트 선택 기준, 평가 지표를 구분해 정리했습니다.

관련 설정과 실험 과정은 [실험일지](docs/experiment_summary.md) 및 [재현성 문서](docs/reproducibility.md)에 정리했습니다.

## 6. 실험을 통해 배운 점

이번 프로젝트에서 가장 중요하게 느낀 점은 **영상 전처리의 효과를 실제 모델 성능으로 검증해야 한다는 것**입니다.

CLAHE는 영상의 대비와 경계 밀도를 높였지만, 탐지 성능에서는 CLAHE+Gamma가 조금 더 나은 결과를 보였습니다. Faster R-CNN에서는 오히려 Original 입력이 더 좋은 경우도 있었습니다.

전처리 방법은 단독으로 평가하기보다 사용하려는 모델과 학습 조건을 함께 고려해야 한다고 생각합니다.

또한 모델 크기를 키우거나 학습을 오래 진행한다고 반드시 성능이 향상되는 것은 아니었습니다.

모델 구조 자체뿐 아니라 작은 병변의 특성, 데이터 품질, 학습 조건, 평가 방식이 모두 영향을 줄 수 있다는 점을 경험했습니다.

### 향후 개선 방향

현재 결과는 지정된 검증 데이터와 실험 조건을 기준으로 한 것으로, 임상 환경에서의 활용 가능성을 검증한 결과는 아닙니다.

앞으로는 환자 단위 데이터 분할과 데이터 누수 점검, 별도 테스트 데이터 평가, 여러 Seed의 반복 실험 등을 통해 결과의 신뢰성을 높일 필요가 있습니다.

또한 작은 병변을 대상으로 한 탐지 성능을 별도로 평가하고, ROI 확대와 2단계 탐지 방식이 실제로 도움이 되는지도 확인해 보고 싶습니다.

---

### 관련 문서

- [실험일지](docs/experiment_summary.md)
- [실험 환경 및 재현성](docs/reproducibility.md)
- [결과 및 그래프 설명](results/README.md)
- [데이터 공개 범위 검토](docs/publication_review.md)

이 저장소에는 원본 의료영상·데이터셋·모델 가중치를 포함하지 않았습니다.
