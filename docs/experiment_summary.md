# 실험별 목적·방법·결과·해석

사용자 제공 ZIP의 노트북 11개와 Python 스크립트 3개를 모두 검토했습니다. 아래 결과는 **저장된 실행 기록** 기준이며 정리 과정에서 재학습한 결과가 아닙니다. 셀은 1-based 원본 위치입니다. 코드 내용, 실행 번호, 저장된 출력은 서로 다른 시점일 수 있으므로 완전한 실행 이력을 단정하지 않습니다.

상태 정의: **완료 기록 확인**은 관련 실행의 완료 출력/지표가 존재한다는 뜻입니다. **실패 기록**은 오류 또는 검증 CHECK 출력이 존재한다는 뜻입니다. **미완료/실행 확인 불가**는 출력이 없거나 필요한 학습·평가 증거가 없다는 뜻입니다. 실행 완료와 성능 개선 성공은 구분합니다.

## 전체 흐름

| 자료 | 목적 | 방법 | 실제 결과·근거 | 상태와 해석 |
|---|---|---|---|---|
| `01_dataset.ipynb` | 데이터 규모·협착 bbox 특성 파악 | COCO 탐색, category count, bbox histogram, 5개 이미지 bbox/polygon 시각화 | cell 15/16: train 1000장, annotation 1625. cell 21: category 26만 1625개. cell 25/31: 평균 면적 4836.93, 중앙값 2782.64 px² | 분석 완료 기록 확인. cell 8의 전체 3000장은 stenosis train 수가 아님. 의료영상 출력 5개는 비공개 보존 |
| `02_preprocessing.ipynb` | 국소 대비·혈관 강조 후보 비교 | CLAHE/Gamma/Frangi/조합/Black-hat; Unsharp/Top-hat 탐색. ROI padding 40으로 육안 비교, CLAHE clip×tile grid 탐색 | cell 40: clip 2.0, tile (4,4), gamma 0.8, sigmas 1..4, kernel 15. cell 44/45: 7종 모두 train=1000, val=200 출력 | 전처리 배치 완료 기록 확인. Unsharp/Top-hat은 7종 배치/YOLO 성능 실험으로 간주하지 않음. 의료영상 포함 시각화 12개 비공개 |
| `03_quality_metrics.ipynb` | 전처리 품질을 정량화 | entropy/std contrast/Canny edges/Laplacian/Brenner, 전체·GT ROI 평균 | cell 5: train 1000개 평가 완료. cell 7: 8개 파이프라인 평균 표·그래프. CLAHE ROI contrast 34.135542, 원본 17.958918 | train 품질 평가 완료 기록 확인. cell 9 rank/composite 계산은 미실행. sharpness 지표가 높아도 노이즈일 수 있음 |
| `03.5_Data Preparation.ipynb` | 전처리/원본 images와 YOLO detect labels 생성 | class 26→0, bbox 정규화, 파이프라인별 복사 | cell 6: 원본 train=1000,val=200. cell 8: train boxes=1625, empty=3; val boxes=406, empty=0. cell 10: 전처리 6종 val 이미지=0, label=200 CHECK | 준비 일부 실패 기록. 초기 경로/복사 셀은 실행 번호 없음. 현재 코드에 있는 Black-hat은 해당 출력표에 없어서 당시 처리 여부 확인 필요. 원본은 OK; 선택 2종은 `04`에서 복구 기록 |
| `03.6_coco_to_yolo_seg.ipynb` | COCO polygon → YOLO instance segmentation labels | converter 생성/import, 통계 확인, split별 label 저장 | 모든 코드 셀의 execution_count=null, outputs 없음 | 실행 확인 불가. markdown의 train/val 기대치는 실제 segmentation 결과가 아님. 실제 mask 검수·학습·평가는 미완료 |
| `04_train_yolo(cursor).ipynb.ipynb` | 전처리 효과를 동일 YOLO11n에서 비교 | selected 3종 이미지 동기화, dataset check, pretrained YOLO11n, 100 epoch, imgsz640,batch16,seed42 | cell 4: CLAHE/CG train1000,val200 복사. cell 5/6: 3종 ready=True. cell 7: 3회 100 epoch 완료 및 best.pt 최종 검증 | 학습 완료 기록 확인. Original→CG mAP50-95 0.121→0.127. 데이터/weights가 없어 새 환경에서 재현 여부 확인 필요 |
| `05A_yolo11_result_aggregation.ipynb` | 전처리 비교표·그래프 집계 | CSV 최고 mAP50-95 epoch 선택 후 `use_manual=True`로 최종 validation 값 override | cell 3: best epoch 74/86/90, mAP50-95 0.12102/0.12320/0.12703. cell 4/5/6: manual 0.121/0.123/0.127, CG 상대+4.96% | 집계 완료 기록 확인. 실제 원본 CSV 미포함. override는 `04` 최종 검증과 대조 완료; 두 표는 선택 방식이 다르므로 따로 보존 |
| `05B_mmdet_swin_convnext.ipynb` | 탐지기/모델 크기 비교 | 실제 구현은 torchvision Faster R-CNN; VOC 11-point AP50. YOLO11n/s 50 epoch도 실행 | cell 7: ResNet50+CG 100 epoch, 최대0.3109045754. cell 8: ResNet50 Original/CG, ResNet-v2 CG 각30 epoch. cell 11: n/s 각50 epoch 완료 | 실행 완료 기록 확인. 파일명과 달리 MMDetection Swin/ConvNeXt 학습 없음. 여러 실행에서 결과 폴더 재사용. matrix에서는 CG가 Original보다 낮음 |
| `05C_yolo11_custom_backbone_skeleton.ipynb` | 외부 backbone 인터페이스 탐색 | ConvNeXt/Swin/ViT 이름의 convolution stub, TinyFPN, YOLO baseline, 랜덤 입력/타깃 MSE toy train | cell 3: feature shape 출력. cell 6: Swin stub toy 출력. cell 8: 실제 YOLO11n baseline 50 epoch 완료, 최종mAP50-95=0.122. cell 9: toy loss 4.5122→2.0322→1.2110 | baseline 완료 기록, stub smoke 기록 확인. 실제 Swin transformer/ConvNeXt/ViT 탐지 모델 통합 아님. 랜덤 데이터 loss는 협착 성능 수치가 아님 |
| `05D_backbone_experiment_matrix.ipynb` | 확장 본실험 matrix 설계 | AdamW/CosineAnnealingLR, ResNet/MobileNet matrix, YOLO·ConvNeXt 옵션 | cell 4: `NameError: name 'random' is not defined`. 다른 코드 셀 outputs 없음 | 실패 기록/본실험 미완료. import 정의 셀 미실행. CG 경로가 `02_clahe_gamma`로 되어 있어 실제 `04_clahe_gamma`와 불일치. negative 이미지 선택도 05B와 다름 |
| `05E_compare_all_results.ipynb` | 여러 모델 결과 한 번에 비교 | FRCNN AP50, YOLO 최고 mAP50 epoch 집계, family leader, pivot/그래프 | cell 4: `NameError: name 'load_frcnn_rows' is not defined`. cell 5~8: 표·그래프 있음. YOLO leader mAP50 0.345780, 같은 epoch mAP50-95 0.12645 | 집계 실패 기록과 후속 출력 공존. 전체 최신 실행 성공 확인 불가. 출력은 보존하되 새 실행 결과로 단정하지 않음. 서로 다른 평가 방식/선택 규칙 주의 |
| `06A_convert_coco_to_yolo_seg.py` | 독립 segmentation 변환기 | category26→0, crowd/RLE/잘못된 polygon 스킵, 좌표 정규화, 파이프라인 mirror | 코드 구현만 제공됨; ARCADE 실행 결과 없음 | 실제 실험 확인 필요. 정리 시 synthetic COCO 기능 검증을 수행했으며 의료 데이터 실험 결과와 구분 |
| `06B_build_roi_crop_pipeline.py` | 작은 병변 확대 학습 데이터 준비 | GT polygon bbox 군집, extent 제한, padding crop, 640 resize, polygon 재정규화 | 실행 로그·생성 이미지·탐지 성능 수치 없음 | 실제 실험 미완료/확인 필요. background를 건너뛰고 GT 기반 crop을 생성하므로 full-image 검증과 직접 비교 불가 |
| `06C_roi_crop_cascade.py` | 추론에서 2단계 확대 검출 | stage1 예측 군집→crop→stage2→좌표 역변환→NMS; TP/FP/FN helper | 모델/실행 로그/평가 수치 없음 | 실제 실험 미완료/확인 필요. stage1 무검출이면 종료하므로 놓친 병변을 stage2가 복구할 수 없음 |

## YOLO 세부 결과

`results/tables/yolo_best_checkpoint_validation.csv`는 다음 로그를 그대로 추출한 파일입니다. Validation은 모두 200장·406 instance입니다.

| 원본·셀 | 실험 | epoch 완료 기록 | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 04 cell 7 | `yolo11n_original` | 100 | 0.364 | 0.387 | 0.320 | 0.121 |
| 04 cell 7 | `yolo11n_clahe` | 100 | 0.381 | 0.390 | 0.320 | 0.123 |
| 04 cell 7 | `yolo11n_clahe_gamma` | 100 | 0.381 | 0.387 | 0.330 | 0.127 |
| 05B cell 11 | `yolo11n__clahe_gamma` | 50 | 0.346 | 0.404 | 0.296 | 0.122 |
| 05B cell 11 | `yolo11s__clahe_gamma` | 50 | 0.356 | 0.401 | 0.310 | 0.121 |
| 05C cell 8 | `yolo11n_Ctrack_clahe_gamma` | 50 | 0.346 | 0.404 | 0.296 | 0.122 |

50 epoch와 100 epoch 조건을 합쳐 모델 크기 효과라고 표현하지 않습니다. 05B와 05C의 n 수치가 같아도 독립 반복 실험의 증거가 되지 않습니다. seed·설정·원본 산출물 확인이 필요합니다.

## Faster R-CNN 세부 결과

epoch별 dict를 파싱한 값입니다. AP50은 자체 구현 VOC 11-point 방식입니다. 표시 값은 읽기 편하게 반올림했으며 정확한 값은 `frcnn_history_summary.csv`와 개별 history에 있습니다.

| 원본·셀 | 모델·입력 | 기록된 epoch | 최고 epoch | 최대 AP50 | 마지막 AP50 |
|---|---|---:|---:|---:|---:|
| 05B cell 7 | ResNet50-FPN / CG 단일 실행 | 100 | 4 | 0.310905 | 0.284617 |
| 05B cell 8 | ResNet50-FPN / Original matrix | 30 | 3 | 0.295686 | 0.255291 |
| 05B cell 8 | ResNet50-FPN / CG matrix | 30 | 4 | 0.290947 | 0.234812 |
| 05B cell 8 | ResNet50-FPN-v2 / CG matrix | 30 | 3 | 0.268508 | 0.216537 |
| 05B cell 9 | MobileNetV3-FPN / CG 집계만 | 확인 필요 | 5(checkpoint metadata) | 0.185888(출력 정밀도) | 확인 필요 |

CG 단일 실행과 matrix는 같은 폴더명을 사용합니다. `05B` cell 9/`05E`는 후속 파일 집계로 CG 값 0.290947을 보여 주며, 이전 100 epoch의 0.310905 기록과 합치지 않습니다. train loss 하락 중 validation AP가 최고치보다 낮아지는 기록은 일반화 문제 검토의 근거지만, 반복실험 없이 과적합의 원인을 확정하지 않습니다.

## 성과와 미해결을 구분한 포트폴리오 설명

- 확인된 개선: 동일 YOLO11n 100 epoch 비교에서 CLAHE+Gamma의 최종 validation mAP50-95가 원본보다 0.006 높습니다.
- 기대와 달랐던 결과: train 영상 대비 개선이 탐지 지표의 같은 폭 개선으로 이어지지 않았습니다. Faster R-CNN matrix에서는 Original이 CG보다 높고, YOLO11s는 50 epoch 조건의 mAP50-95에서 n을 넘지 못했습니다.
- 실행 실패: 전처리 val 이미지 누락, 05D/05E 선행 정의 셀 NameError. val 누락은 04의 선택 두 파이프라인에 한해 복구 기록이 있습니다.
- 설계/미완료: 정식 Swin/ConvNeXt/ViT/InternImage 탐지 모델, segmentation 실제 실행/학습, ROI-crop/cascade 성능 검증.
- 확인 필요: 데이터 권한·정확한 환경·원본 CSV/weights·환자 단위 split·test 평가·여러 seed·통일된 평가 프로토콜.
