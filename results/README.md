# 저장된 결과의 출처

모든 파일은 원본 노트북의 **저장된 출력에서 추출**했습니다. 이 정리 작업에서 의료 데이터로 모델을 실행하지 않았습니다. 원본 `history.csv`, `results.csv`, weights는 ZIP에 없으며, 이 폴더의 CSV는 저장된 표/epoch 로그를 추출한 결과입니다. 원래 학습 산출물 파일을 복원했다고 표현하지 않습니다.

## Figures

| 파일 접두사·셀 | 내용 | 해석 시 주의 |
|---|---|---|
| `01_dataset__cell_023` | bbox 면적 histogram | train 집계 그래프 |
| `03_quality_metrics__cell_007` | 전처리 품질 비교 | 이미지 지표를 탐지 성능으로 해석하지 않음 |
| `05A_yolo11_result_aggregation__cell_006` | YOLO11n 전처리 비교 | 수동 override 이후 그래프. `04` 최종 검증 로그와 대조 확인 |
| `05E_compare_all_results__cell_006` | 전체 모델 AP50 비교 | VOC/Ultralytics 평가 방식 혼재, 반복 실행의 같은 라벨이 겹칠 수 있음 |
| `05E_compare_all_results__cell_007` | YOLO mAP50-95 비교 | 최고 mAP50 epoch에서 얻은 mAP50-95. 최고 mAP50-95나 best.pt 검증 값과 다름 |

원본의 `image/png` base64를 디코딩하여 그대로 저장했습니다. 그래프를 다시 그리거나 숫자를 변경하지 않았습니다. 집계 PNG 5개는 정리된 노트북에도 유지합니다.

의료영상 17개(원본 예시 5개, 전처리/ROI/튜닝 시각화 12개)는 **공개 저장소에서 제외**하고 로컬에 보존했습니다. `image_manifest.json`에는 22개 모두의 원본 노트북·셀·출력 번호와 SHA-256이 있습니다. 셀 번호는 1부터 세는 원본 셀 위치이며 정리본에서도 같습니다.

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
