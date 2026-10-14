# 검토용 변경 보고서

## 원본과 저장소 비교

- 기존 저장소 baseline: `a01551494c9896804aa9fe86ad888d24b73a7cae`(원격 main과 일치했던 체크아웃), 파일은 README.md와 .gitignore 두 개.
- 제공 파일의 실제 이름은 `yolo.zip`입니다. 요청의 `yolo(1).zip`과 이름이 다르지만 이번 분석은 이 첨부 파일 기준입니다.
- ZIP SHA-256: `6ecb0af2071de9ee5852891ade8ce4d1b21fa3f25b07550274c3924d0fa11f50`.
- 내부 파일은 Jupyter Notebook **11개**, Python script **3개**. CRC 검사 통과. 동일 내용(hash)의 중복 파일은 없습니다.
- 노트북에 표시된 다른 파일(05F 등)·weights·학습 산출물은 이 ZIP에는 없어 분석 완료로 간주하지 않았습니다.
- 로컬 작업 브랜치: `portfolio/xca-yolo-review`. 원본 ZIP과 추출본은 보존했으며 기존 저장소 파일 삭제가 없습니다. 원격 push·PR 생성·main 반영·병합은 수행하지 않았습니다.

## 변경 파일 목록

| 구분 | 파일·범위 | 작업 |
|---|---|---|
| 수정 | `README.md` | 한국어 포트폴리오 설명: 목적, 데이터, 전처리, 모델 비교, 평가, 한계, 실행 절차, 확인 필요 사항 |
| 수정 | `.gitignore` | 기존 규칙 유지, 원본 ZIP/데이터/annotations/weights/개인 설정/학습 산출물 제외 규칙 추가 |
| 추가 | `requirements.txt` | 실제 import 기반 의존성 목록. 미확인 버전을 임의로 고정하지 않음 |
| 추가 | `notebooks/`의 원본 11개 | 파일명/셀 순서/실행 번호 유지. 경로 수정·출력 정리·비공개 의료영상 제외 |
| 추가 | `scripts/06A_convert_coco_to_yolo_seg.py` | 원본 변환기 유지, 기본 데이터 경로만 프로젝트 기준으로 변경 |
| 추가 | `scripts/06B_build_roi_crop_pipeline.py` | 원본과 byte-identical |
| 추가 | `scripts/06C_roi_crop_cascade.py` | 원본과 byte-identical |
| 추가 | `scripts/verify_portfolio.py` | 데이터·학습·네트워크 없이 결과/코드/이미지 보존을 검증하는 도구 |
| 추가 | `results/figures/` 5개 PNG | 원본 집계 그래프만 디코딩해 byte-identical 보존 |
| 추가 | `results/tables/` | 원본 11개 HTML table와 CSV 각각, FRCNN history 4개+요약, YOLO 최종검증 CSV. 총 17개 CSV+11개 HTML |
| 추가 | `results/evidence/` 11개 TXT | 지표/완료/오류 기록, 개인 경로 익명화. 과거 저장된 출력임을 명시 |
| 추가 | `results/README.md`, `results/image_manifest.json` | 수치 출처·선택 규칙 차이, 전체 22개 이미지 공개 여부와 hash |
| 추가 | `docs/experiment_summary.md` | 원본 14개별 목적·방법·결과·해석, 성공/실패/미완료 구분 |
| 추가 | `docs/reproducibility.md` | 과거 환경 근거, 실행 순서, 미해결 코드 문제·재현성 영향 |
| 추가 | `docs/publication_review.md` | 의료영상/개별 annotation/경로/비밀정보 검수, 로컬 원본 보존 |
| 추가 | `docs/change_report.md` | 이 보고서 |
| 추가 | `docs/source_manifest.json` | 원본 ZIP·14개 파일 및 정리본 hash, notebook 출력 개수 |
| 추가 | `docs/code_changes.json` | 변경된 코드 8건의 셀 위치·전후 코드·hash. 변경 전 개인 경로 가림 |
| 추가 | `docs/output_changes.json` | 공개 복사본의 출력 정리 내역. 원본 전체 output 객체는 비공개 보존 |

정리본에 포함되는 모든 파일의 정확한 이름은 Git diff 또는 별도 검토용 patch로 확인할 수 있습니다. 검증용 가상환경·synthetic fixtures·원본/개인 출력 archive는 Git 저장소 밖에 있습니다.

## 재현성에 영향을 주는 수정

실행 source 변경은 **8건**입니다. Notebook source 7개 셀과 06A 1개 파일에 한정합니다.

| 위치 | 변경 | 영향 |
|---|---|---|
| 01 cell4 | 개인 절대 PROJECT_ROOT → cwd 부모 | `notebooks/`에서 실행 필요 |
| 03.5 cell2 | 개인 DATA_DIR → cwd 부모/data | 동일 데이터 디렉터리 구조 필요 |
| 03.6 cell3 | 개인 PROJECT_DIR → cwd 부모 | 동일 작업 디렉터리 전제 |
| 03.6 cell7 | 기존 converter 파일이 있으면 쓰지 않음 | 원본 독립 script 보존. 없을 때만 기존 embedded converter 생성 |
| 05B cell2 | 개인 서버 fallback 제거 | 프로젝트의 data 입력을 사용하며 과거 서버를 자동 검색하지 않음 |
| 05D cell2 | 개인 서버 fallback 제거 | cwd 부모 기준. 기존 CG 경로 불일치는 수정하지 않음 |
| 05E cell2 | 개인 서버 fallback 제거 | cwd 부모 기준. 오류/평가 방식은 그대로 |
| 06A | 개인 기본 data path → script 기준 프로젝트/data | `--data-dir`로 override 가능, converter 알고리즘 불변 |

추가로 03.6 셀 13개에 누락된 ID metadata를 부여했습니다. 모든 원본 markdown/source의 나머지 부분과 셀 순서/실행 번호는 원본 대조 검증을 통과했습니다. 모델 구조·하이퍼파라미터·평가/라벨/crop 알고리즘을 변경하지 않았습니다.

의료영상 17개와 개별 annotation 출력은 로컬 보존 후 공개 복사본에서 제외했습니다. 전체 원본 출력을 삭제하지 않았습니다. 진행률 widget·반복 training boilerplate·warning·디렉터리 나열은 공개 출력만 정리했으며 핵심 수치·표·오류·190개 FRCNN epoch history를 추출 보존했습니다.

## 검증 결과

- ZIP hash·CRC, 14개 원본 file hash: 통과. 원본 ZIP 변경 없음.
- 11개 notebook nbformat validation: 통과, 경고 0. Python script 4개 및 notebook 일반 Python 셀 syntax 검증: 통과(install magic 셀은 AST 검사 제외).
- 모든 원본 code-cell output JSON 비교, 22개 원본 PNG hash, 공개 PNG 5개 hash, 문서화된 source 수정 8건 외 셀 보존, execution_count/셀 순서: 통과.
- YOLO 최종검증 CSV 6개 행, validation200/instance406, 04 핵심 3개 mAP50-95 값, FRCNN history 190개 연속 epoch 행: 통과.
- **합성 데이터 테스트 10개 실행·10개 통과**: schema/syntax, 원본 보존, polygon validation, category/crowd/RLE/negative 변환, CLI 기본/override 경로, 기존 converter 덮어쓰기 방지, cluster extent/crop clamp, 실제 crop PNG·label 출력 및 dry-run/원본 불변, label remap, 06C 순수 IoU/TP·FP·FN helper.
- 검증 환경: 별도 Python3.12.14 가상환경, opencv-python-headless5.0.0.93, nbformat5.11.1. 이 환경의 성공을 과거 전체 모델 환경 재현으로 표현하지 않습니다.
- 공개 후보 텍스트의 개인 home path/GitHub token/AWS key/private key/email/credential assignment pattern 검사: 발견 없음. 의료 이미지 17개가 공개 notebook/results에 없는지 검증했습니다.
- README/문서의 local link 검증: 누락 없음. 공개 대상 파일을 명시적으로 선택해 패키징했습니다.
- **미실행**: 실제 ARCADE 데이터 변환/전처리/재학습, torch/torchvision NMS runtime, cascade 두 모델 통합, GPU 실행, 정확한 전체 requirements 설치·과거 환경 복원, 원격 push·PR.

06C는 torch runtime 없이 AST로 분리한 **순수 평가 helper만** 합성 테스트했습니다. 모델 추론·NMS·2단계 cascade 실행 성공을 주장하지 않습니다. 합성 테스트 10개가 기존 의료 실험의 결과를 추가하지는 않습니다.

## 확인이 필요한 사항

1. 데이터 공식 출처·버전·라이선스, 의료영상/파생 이미지의 공개 조건, 환자 단위 split 및 비식별화 근거.
2. 과거 전체 패키지 버전/설치 출처·CUDA 드라이버, 원본 CSV/args.yaml/weights 확보 가능 여부.
3. 05B MobileNet의 전체 학습 기록, 원본 ZIP에 없는 05F 등 추가 실험 자료.
4. 05D/05E 전체 재실행, 잘못된 CG 경로·LR 미반영·stub channel·평가 규칙·반복 실행 덮어쓰기 문제의 별도 수정.
5. test/external validation, 여러 seed, 신뢰구간, 통일된 evaluator, cascade의 실제 개선 검증.
6. 브랜치 push 및 PR 생성 승인. 승인 후 검토용 PR을 만들고 main 병합은 사용자 검토에 맡깁니다.

## 전체 변경 파일 이름

기존 README.md/.gitignore는 수정, 나머지는 추가입니다. 파일 삭제는 없습니다.

- `.gitignore`
- `README.md`
- `docs/change_report.md`
- `docs/code_changes.json`
- `docs/experiment_summary.md`
- `docs/output_changes.json`
- `docs/publication_review.md`
- `docs/reproducibility.md`
- `docs/source_manifest.json`
- `notebooks/01_dataset.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03.5_Data Preparation.ipynb`
- `notebooks/03.6_coco_to_yolo_seg.ipynb`
- `notebooks/03_quality_metrics.ipynb`
- `notebooks/04_train_yolo(cursor).ipynb.ipynb`
- `notebooks/05A_yolo11_result_aggregation.ipynb`
- `notebooks/05B_mmdet_swin_convnext.ipynb`
- `notebooks/05C_yolo11_custom_backbone_skeleton.ipynb`
- `notebooks/05D_backbone_experiment_matrix.ipynb`
- `notebooks/05E_compare_all_results.ipynb`
- `requirements.txt`
- `results/README.md`
- `results/evidence/01_dataset.txt`
- `results/evidence/02_preprocessing.txt`
- `results/evidence/03.5_Data_Preparation.txt`
- `results/evidence/03.6_coco_to_yolo_seg.txt`
- `results/evidence/03_quality_metrics.txt`
- `results/evidence/04_train_yolo_cursor.txt`
- `results/evidence/05A_yolo11_result_aggregation.txt`
- `results/evidence/05B_mmdet_swin_convnext.txt`
- `results/evidence/05C_yolo11_custom_backbone_skeleton.txt`
- `results/evidence/05D_backbone_experiment_matrix.txt`
- `results/evidence/05E_compare_all_results.txt`
- `results/figures/01_dataset__cell_023_output_01.png`
- `results/figures/03_quality_metrics__cell_007_output_02.png`
- `results/figures/05A_yolo11_result_aggregation__cell_006_output_01.png`
- `results/figures/05E_compare_all_results__cell_006_output_01.png`
- `results/figures/05E_compare_all_results__cell_007_output_01.png`
- `results/image_manifest.json`
- `results/tables/03_quality_metrics__cell_007_output_03.csv`
- `results/tables/03_quality_metrics__cell_007_output_03.html`
- `results/tables/04_train_yolo_cursor__cell_005_output_01.csv`
- `results/tables/04_train_yolo_cursor__cell_005_output_01.html`
- `results/tables/04_train_yolo_cursor__cell_007_output_02.csv`
- `results/tables/04_train_yolo_cursor__cell_007_output_02.html`
- `results/tables/04_train_yolo_cursor__cell_008_output_01.csv`
- `results/tables/04_train_yolo_cursor__cell_008_output_01.html`
- `results/tables/05A_yolo11_result_aggregation__cell_003_output_01.csv`
- `results/tables/05A_yolo11_result_aggregation__cell_003_output_01.html`
- `results/tables/05A_yolo11_result_aggregation__cell_004_output_01.csv`
- `results/tables/05A_yolo11_result_aggregation__cell_004_output_01.html`
- `results/tables/05A_yolo11_result_aggregation__cell_005_output_02.csv`
- `results/tables/05A_yolo11_result_aggregation__cell_005_output_02.html`
- `results/tables/05B_cell_007__single_resnet50_fpn__clahe_gamma_history.csv`
- `results/tables/05B_cell_008__frcnn_resnet50_fpn__clahe_gamma_history.csv`
- `results/tables/05B_cell_008__frcnn_resnet50_fpn__original_history.csv`
- `results/tables/05B_cell_008__frcnn_resnet50_fpn_v2__clahe_gamma_history.csv`
- `results/tables/05B_mmdet_swin_convnext__cell_009_output_02.csv`
- `results/tables/05B_mmdet_swin_convnext__cell_009_output_02.html`
- `results/tables/05E_compare_all_results__cell_005_output_02.csv`
- `results/tables/05E_compare_all_results__cell_005_output_02.html`
- `results/tables/05E_compare_all_results__cell_005_output_04.csv`
- `results/tables/05E_compare_all_results__cell_005_output_04.html`
- `results/tables/05E_compare_all_results__cell_008_output_02.csv`
- `results/tables/05E_compare_all_results__cell_008_output_02.html`
- `results/tables/frcnn_history_summary.csv`
- `results/tables/yolo_best_checkpoint_validation.csv`
- `scripts/06A_convert_coco_to_yolo_seg.py`
- `scripts/06B_build_roi_crop_pipeline.py`
- `scripts/06C_roi_crop_cascade.py`
- `scripts/verify_portfolio.py`
