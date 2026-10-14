# 저장된 결과의 출처

기존 파일은 원본 노트북의 **저장된 출력에서 추출**했습니다. 아래에 구분한 새 그래프는 그 공개 CSV를 다시 시각화한 결과입니다. 의료 데이터로 새 학습이나 추론은 수행하지 않았습니다. 원본 `history.csv`, `results.csv`, weights는 ZIP에 없으며, 이 폴더의 CSV는 저장된 표/epoch 로그를 추출한 결과입니다. 원래 학습 산출물 파일을 복원했다고 표현하지 않습니다.

## Figures

| 파일 접두사·셀 | 내용 | 해석 시 주의 |
|---|---|---|
| `01_dataset__cell_023` | bbox 면적 histogram | train 집계 그래프 |
| `03_quality_metrics__cell_007` | 전처리 품질 비교 | 이미지 지표를 탐지 성능으로 해석하지 않음 |
| `05A_yolo11_result_aggregation__cell_006` | YOLO11n 전처리 비교 | 수동 override 이후 그래프. `04` 최종 검증 로그와 대조 확인 |
| `05E_compare_all_results__cell_006` | 전체 모델 AP50 비교 | VOC/Ultralytics 평가 방식 혼재, 반복 실행의 같은 라벨이 겹칠 수 있음 |
| `05E_compare_all_results__cell_007` | YOLO mAP50-95 비교 | 최고 mAP50 epoch에서 얻은 mAP50-95. 최고 mAP50-95나 best.pt 검증 값과 다름 |

위 원본 그래프 5개는 `image/png` base64를 디코딩하여 그대로 저장했습니다. 이 파일들은 다시 그리거나 숫자를 변경하지 않았고, 정리된 노트북에도 유지합니다. 새로 그린 그래프는 다음 절에 별도로 구분합니다.

의료영상 17개(원본 예시 5개, 전처리/ROI/튜닝 시각화 12개)는 **공개 저장소에서 제외**하고 로컬에 보존했습니다. `image_manifest.json`에는 22개 모두의 원본 노트북·셀·출력 번호와 SHA-256이 있습니다. 셀 번호는 1부터 세는 원본 셀 위치이며 정리본에서도 같습니다.

## CSV로 다시 그린 README용 그래프

기존 그래프를 먼저 비교한 뒤 README에는 필요한 자료만 배치했습니다. 원본 5개를 지우거나 덮어쓰지 않았으며, README에서 같은 내용을 보여 주는 원본·새 그래프를 중복 배치하지 않았습니다.

| README 위치 | 사용 파일 | 선택 이유·입력 |
|---|---|---|
| 프로젝트 배경 | [기존 bbox 분포](figures/01_dataset__cell_023_output_01.png) | 분포와 축을 읽을 수 있어 재사용. 개별 bbox 데이터가 없으므로 새 histogram을 만들지 않음 |
| 전처리 실험 | [preprocessing_roi_quality.png](figures/preprocessing_roi_quality.png) | 기존 8개 입력 그래프에서 학습 비교에 사용한 3개만 골라 단위·수치를 명시. [품질 평균 CSV](tables/03_quality_metrics__cell_007_output_03.csv) 사용 |
| YOLO 학습 결과 | [yolo11n_final_validation.png](figures/yolo11n_final_validation.png) | 기존 4개 지표 그래프 대신 mAP50·mAP50-95에 집중. [최종 검증 CSV](tables/yolo_best_checkpoint_validation.csv)의 `yolo11n_original`, `yolo11n_clahe`, `yolo11n_clahe_gamma` 100 epoch 행만 사용 |
| 추가 모델 비교 | [frcnn_ap50_history.png](figures/frcnn_ap50_history.png) | 기존 최대값 비교에 없던 epoch 변화 추가. ResNet50-FPN의 30 epoch Original/CG와 별도 100 epoch CG를 두 패널로 구분 |

기존 품질 그래프에는 전체 8개 입력과 global Laplacian variance가 이미 있어 같은 내용을 새 그림으로 추가하지 않았습니다. Precision·Recall은 README 표에 남겨 별도 그래프를 중복 생성하지 않았습니다. ResNet50-FPN-v2와 MobileNet 수치는 기존 CSV·실험일지에 유지하고, 학습 이력 그림은 같은 backbone의 입력 비교에 집중했습니다.

### 지표와 표시 단위

- **ROI Contrast:** 8-bit 밝기 값(0–255)의 표준편차이며 표시 단위는 gray level입니다. 검출 AP나 물리적 대비 단위가 아닙니다.
- **ROI Edge Density:** ROI 픽셀 중 Canny 경계 픽셀 비율입니다. 원본 ratio에 100을 곱해 %로 표시했습니다. Canny threshold는 50/150이며, 정답 bbox ROI의 이미지별 평균 방식은 실험일지에 설명되어 있습니다.
- **YOLO mAP50 / mAP50-95:** Ultralytics의 AP ratio입니다. 최종 `best.pt` 검증 로그의 반올림 수치를 그대로 사용하며 `05A` 최고 epoch 집계나 `05E` 최고 mAP50 집계는 입력으로 사용하지 않습니다. 패널별 축 범위는 다르지만 모두 0부터 시작합니다.
- **Faster R-CNN AP50:** 직접 구현한 VOC 11-point AP, IoU 0.50·score ≥ 0.05입니다. 원본 epoch 값을 보간·평활화 없이 연결했습니다. 두 패널의 y축 범위는 같습니다. 최고값이 같은 epoch가 여럿이면 최초 epoch를 표시합니다.

막대의 표시 소수는 품질 지표 2자리, YOLO 3자리이고 학습 이력 표는 6자리로 반올림했습니다. 막대 높이와 선 좌표는 CSV 정밀도를 사용합니다. 큰 영상 품질 지표를 더 높은 탐지 성능으로 해석하거나, 작은 병변의 별도 AP 개선을 입증했다고 표현하지 않습니다.

### Faster R-CNN 그림의 정확한 입력

| 패널·입력 | 원본 CSV | 최고 AP50 / epoch | 마지막 AP50 |
|---|---|---|---|
| 30 epoch / Original | [cell 008 Original](tables/05B_cell_008__frcnn_resnet50_fpn__original_history.csv) | 0.2956856003735381 / 3 | 0.25529051900250077 |
| 30 epoch / CLAHE+Gamma | [cell 008 CG](tables/05B_cell_008__frcnn_resnet50_fpn__clahe_gamma_history.csv) | 0.2909470683437324 / 4 | 0.23481221908034494 |
| 100 epoch / CLAHE+Gamma | [cell 007 CG](tables/05B_cell_007__single_resnet50_fpn__clahe_gamma_history.csv) | 0.3109045753710932 / 4 | 0.2846165126854997 |

최고값과 마지막 값은 각 history에서 계산해 [기존 요약 CSV](tables/frcnn_history_summary.csv)와 대조합니다. 그래프의 범례에서 ◆ Best는 최초 최고값 epoch, ● Last는 마지막 epoch를 뜻하며, 아래 요약표에서도 값과 최고 epoch를 확인할 수 있습니다. 30/100 epoch는 서로 다른 실행이며 같은 폴더를 재사용했던 기록을 합치지 않았습니다. Loss를 새로 추정하거나 추가하지 않았습니다.

100 epoch 실행의 최고값은 epoch 4에 나타났지만, 이 사실만으로 과적합이나 성능 정체의 원인을 확정하지 않습니다. 모델·학습 조건과 반복 실험을 함께 확인해야 합니다.

### 재생성

현재 환경의 Python 3.12.14, Matplotlib 3.10.8에서 기존 설치 패키지로 생성했습니다. 패키지를 추가 설치하거나 기존 requirements를 변경하지 않았습니다. 스크립트는 표준 라이브러리 CSV 로더와 Matplotlib만 사용하며 작업 디렉터리에 의존하지 않습니다.

```bash
python scripts/plot_experiment_results.py
# 별도 위치에서 결과를 확인하려면:
python scripts/plot_experiment_results.py --output-dir /tmp/xca-plots
```

기본 출력은 새 PNG 3개와 [visualization_manifest.json](figures/visualization_manifest.json)입니다. 기본 240 DPI로 품질·YOLO 그림은 2832×1032, 학습 이력은 2976×1392 픽셀입니다. Original은 청회색, CLAHE는 갈색, CLAHE+Gamma는 녹색으로 통일했습니다. 축에는 단위를 남기고, 작은 부제·출처·해석 문구는 이 문서로 모았습니다. README의 그림은 bbox를 포함한 4개 그대로이며, 학습 이력 PNG의 세로 길이는 이전 1872픽셀에서 1392픽셀로 줄였습니다.

Manifest에는 입력 CSV hash, 실제로 그린 값/epoch, 색상, 표시 단위, 소프트웨어 버전, 생성 PNG hash, 재사용 bbox PNG hash가 있습니다. 원본을 추출한 `image_manifest.json`의 범위는 기존 22개 출력 그대로이며 새 그림의 검증 정보와 구분합니다. Matplotlib/폰트 캐시는 임시 디렉터리를 사용하며 실험 코드·CSV·원본 그림은 수정하지 않습니다.

## Tables와 evidence

- `yolo_best_checkpoint_validation.csv`: `04` cell 7(3개), `05B` cell 11(2개), `05C` cell 8(1개)의 **최종 best.pt 검증 로그**. 소수 정밀도는 원본 로그 그대로입니다.
- `05A_*cell_003*.csv`: 원본 `results.csv`를 읽었던 **최고 mAP50-95 epoch** 표 출력. Original / CLAHE / CLAHE+Gamma의 mAP50-95는 0.12102 / 0.12320 / 0.12703이며 epoch 74 / 86 / 90입니다. 원본 CSV 자체는 없습니다.
- `05A_*cell_004*.csv`, `cell_005`: 수동 override와 그 수치로 계산한 delta 표. 앞선 CSV 최고 epoch의 Precision/Recall과 차이가 있습니다. 별도 파일로 보존해 구분합니다.
- `03_quality_metrics*`: train 전체 8개 파이프라인의 평균 지표 표. 이미지별 전체 원본 CSV는 ZIP에 없습니다.
- `05B_cell_007*history.csv`: 100개 epoch의 원본 dict를 파싱. `cell_008*history.csv`: 3개 matrix 실행의 30개 epoch씩. `frcnn_history_summary.csv`에 각각의 최대 AP50을 별도로 저장했습니다.
- `05B_*cell_009*`: 이후 폴더 집계 결과. MobileNet은 이 표에만 있어 학습 로그/총 epoch를 확인할 수 없습니다.
- `05E_*`: family/model leader와 pivot 출력. 최고 mAP50 선택이며 05A와 선택 기준이 다릅니다. `05E` cell 4의 NameError도 evidence에 남아 있습니다.
- `evidence/*.txt`: 개인 경로를 `<PROJECT_ROOT>`로 익명화한 지표·학습 완료·오류 기록. progress widget/반복 진행률/일부 디렉터리 나열은 정리했으며 전체 원본 output 객체는 로컬에 따로 보존했습니다.

HTML 표의 내용에서 CSV를 추출했습니다. DataFrame index(첫 열)·빈 헤더는 원본 출력 모양을 따라 유지하며, 소수 정밀도가 표 출력 수준으로 제한됩니다. `...`, `NaN`, 빈 값은 원래의 생략/누락으로 유지합니다. 미출력 값을 추측해 채우지 않았습니다.

## 수치를 합치지 않는 이유

`04` 최종 검증의 mAP50 0.330, `05A` CSV 최고 mAP50-95 epoch의 mAP50 0.32762, `05E` 최고 mAP50 epoch의 mAP50 0.345780은 서로 다른 선택/검증 결과입니다. `05E`의 mAP50-95 0.12645를 `05A`의 0.12703과 같은 epoch의 값처럼 묶지 않습니다.

Faster R-CNN 단일 100 epoch 최대 AP50 0.3109045753710932와 matrix 30 epoch 최대 AP50 0.2909470683437324는 같은 폴더명을 사용한 서로 다른 실행 기록입니다. `exist_ok=True` 및 고정 output 폴더 때문에 이전 파일을 덮어쓸 위험이 있습니다. 구분한 epoch history가 비교 근거이며, 원본 checkpoint의 현재 내용은 확인할 수 없습니다.
