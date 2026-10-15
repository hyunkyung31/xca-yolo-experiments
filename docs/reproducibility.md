# 실행 환경 및 재현성

이 문서는 ARCADE 데이터셋을 이용한 관상동맥 협착 탐지 실험의 실행 환경, 데이터 구조, 학습 설정 및 재실행 시 주의할 사항을 정리한 문서입니다.

실험에는 영상 전처리, YOLO11 학습, Faster R-CNN 비교와 Segmentation·ROI 기반 탐지 파이프라인 구현이 포함되어 있습니다.

일부 실험은 학습과 평가까지 완료했지만, 확장 실험 중에는 코드 구현이나 실행 준비 단계에 머문 작업도 있습니다. 따라서 저장된 성능 결과와 재실행을 통해 확인할 수 있는 범위를 구분했습니다.

## 1. 실험 환경

원본 노트북의 실행 환경과 로그에 나타난 주요 라이브러리 버전은 다음과 같습니다.

| 구성 | 버전 및 환경 |
|---|---|
| Python | 3.12.6 |
| Ultralytics | 8.4.110 |
| PyTorch | 2.13.0+cu130 |
| CUDA Runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 5090 (32111 MiB) |
| scikit-image | 0.26.0 |
| seaborn | 0.13.2 |

위 정보는 실험 당시의 노트북 메타데이터와 실행 로그를 기준으로 합니다. 모든 라이브러리의 정확한 버전이나 설치 조합이 확보된 것은 아닙니다.

OpenCV, torchvision, NumPy, pandas, Pillow, matplotlib 등은 사용했지만 세부 버전이 명확하지 않은 항목도 있습니다.

`requirements.txt`에는 코드에서 사용한 주요 패키지를 정리했습니다. 다만 전체 의존성을 고정한 lockfile은 아니므로 과거 실험 환경과 완전히 동일한 실행 결과를 보장하지 않습니다.

특히 PyTorch, torchvision 및 CUDA는 사용 중인 장비와 드라이버에 맞는 호환 버전을 확인해야 합니다.

## 2. 데이터 구조

실험에는 ARCADE 데이터셋의 협착 탐지 자료를 사용했습니다.

- [ARCADE 공식 사이트](https://arcade.grand-challenge.org/)
- [데이터 배포 페이지](https://zenodo.org/records/7981245)
- [데이터 접근 안내](https://arcade.grand-challenge.org/registration/)

원본 의료영상과 annotation 데이터셋은 저장소에 포함하지 않았습니다. 재실행하려면 데이터 이용 조건을 확인하고 별도로 준비해야 합니다.

프로젝트에서 사용하는 주요 디렉터리 구조는 다음과 같습니다.

```text
data/
├── arcade/
│   └── arcade/
│       └── stenosis/
│           ├── train/
│           │   ├── images/
│           │   └── annotations/train.json
│           └── val/
│               ├── images/
│               └── annotations/val.json
│
├── preprocessed/
│   └── <pipeline>/
│       ├── train/
│       └── val/
│
└── yolo/
    └── <pipeline>/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

`data/preprocessed/`는 전처리 이미지를 저장하는 경로이고, `data/yolo/`는 실제 YOLO 학습에 사용하는 이미지와 라벨을 구성하는 경로입니다.

두 디렉터리의 역할이 다르므로 전처리 이미지 생성 후 학습 입력 구성까지 확인해야 합니다.

노트북의 상대경로는 `notebooks/` 디렉터리를 현재 작업 위치로 사용하는 구성을 기준으로 합니다.

## 3. 실행 환경 준비

프로젝트 루트에서 Python 가상환경을 생성합니다.

```bash
python -m venv .venv
```

운영체제에 따라 가상환경을 활성화합니다.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

CPU 또는 GPU 환경에 적합한 PyTorch 및 torchvision 설치 조합을 먼저 확인한 뒤 필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

저장소 점검 스크립트를 실행할 수 있습니다.

```bash
python scripts/verify_portfolio.py
```

이어서 노트북 디렉터리에서 JupyterLab을 실행합니다.

```bash
cd notebooks
jupyter lab
```

이 명령어들은 재실행을 위한 기본 절차입니다. 실제 ARCADE 데이터를 이용해 모든 실험을 처음부터 끝까지 재실행한 결과는 아닙니다.

## 4. 실험 진행 순서

### 4.1 데이터 분석

**노트북:** `01_dataset.ipynb`

COCO 형식의 annotation을 불러와 이미지 수와 협착 영역의 분포를 분석합니다.

주요 작업은 다음과 같습니다.

- 이미지 및 annotation 개수 확인
- Category 분포 분석
- Bounding Box 크기와 면적 분석
- 원본 이미지에 bbox 및 polygon 시각화

원본 노트북에는 OpenCV 재설치 명령이 포함되어 있습니다. 실행 환경을 변경할 수 있으므로 현재 설치 상태를 먼저 확인해야 합니다.

### 4.2 이미지 전처리

**노트북:** `02_preprocessing.ipynb`

CLAHE, Gamma Correction, Frangi Filter 등의 전처리를 적용합니다.

실험에 사용한 주요 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---|
| CLAHE clipLimit | 2.0 |
| CLAHE tileGridSize | (4, 4) |
| Gamma | 0.8 |
| Frangi sigmas | 1~4 |
| Frangi gamma | 15 |
| Black-hat kernel | 15 |

각 전처리에 대해 학습 이미지 1,000장과 검증 이미지 200장을 생성했습니다.

생성된 이미지는 `data/preprocessed/`에 저장됩니다.

### 4.3 영상 품질 평가

**노트북:** `03_quality_metrics.ipynb`

전처리 영상의 품질을 비교하기 위해 Contrast, Edge Density, Laplacian Variance 등을 계산합니다.

영상 전체와 병변 ROI를 구분해 평가했습니다.

정답 병변이 없는 이미지는 ROI 지표에 0으로 포함했으며, 한 이미지에 병변이 여러 개 있을 경우 병변별 지표를 평균한 뒤 이미지 단위로 집계했습니다.

일부 시각화 예제는 랜덤 선택에 고정 Seed가 없어 재실행 시 선택되는 이미지가 달라질 수 있습니다.

### 4.4 YOLO 학습 데이터 준비

**노트북:** `03.5_Data Preparation.ipynb`

COCO 형식의 Bounding Box를 YOLO 형식으로 변환하고, 학습에 필요한 이미지와 라벨을 구성합니다.

- 협착 Category 26을 YOLO Class 0으로 변환
- Bounding Box 좌표 정규화
- Train / Validation 분리
- 이미지와 라벨 파일 개수 확인
- 전처리별 학습 폴더 구성

초기 작업에서는 전처리 검증 이미지가 YOLO 학습 폴더에 복사되지 않아 검증 이미지 개수가 0으로 나타나는 문제가 있었습니다.

이후 CLAHE와 CLAHE+Gamma 이미지가 각각 학습 1,000장, 검증 200장으로 준비되도록 동기화했습니다.

학습 전 이미지와 라벨의 개수를 반드시 확인해야 합니다.

### 4.5 YOLO11n 학습

**노트북:** `04_train_yolo(cursor).ipynb.ipynb`

Original, CLAHE, CLAHE+Gamma 입력을 이용해 YOLO11n을 학습합니다.

| 항목 | 설정 |
|---|---|
| 모델 | YOLO11n Pretrained |
| Epoch | 100 |
| Image Size | 640 |
| Batch Size | 16 |
| Device | 0 (GPU) |
| Workers | 8 |
| Patience | 30 |
| Seed | 42 |

CPU 환경에서 실행할 경우 `device`를 `cpu`로 변경해야 합니다.

Batch Size와 Workers 또한 시스템 메모리와 장비 성능에 맞게 조정할 수 있지만, 값을 바꾸면 기존 실험과 실행 조건이 달라집니다.

검증에는 이미지 200장과 협착 annotation 406개를 사용했습니다.

| 입력 | mAP50 | mAP50-95 |
|---|---:|---:|
| Original | 0.320 | 0.121 |
| CLAHE | 0.320 | 0.123 |
| CLAHE+Gamma | 0.330 | 0.127 |

위 수치는 각 실험의 `best.pt`를 이용한 최종 검증 결과입니다.

### 4.6 YOLO 성능 집계

**노트북:** `05A_yolo11_result_aggregation.ipynb`

학습 결과를 집계할 때 `use_manual` 설정을 확인해야 합니다.

```python
use_manual = False
```

`True`인 경우 기존에 입력한 수치를 사용하므로 새로운 실험 결과를 집계하려면 해당 옵션을 변경해야 합니다.

최고 mAP50 epoch, 최고 mAP50-95 epoch, 최종 체크포인트 검증 결과는 서로 다른 기준이므로 구분해 비교합니다.

### 4.7 Faster R-CNN 및 YOLO 크기 비교

**노트북:** `05B_mmdet_swin_convnext.ipynb`

torchvision 기반 Faster R-CNN을 이용해 모델별 성능을 비교했습니다.

주요 실험은 ResNet50-FPN, ResNet50-FPN-v2 및 MobileNetV3-FPN을 대상으로 구성했습니다.

평가에는 VOC 11-point AP50를 사용했습니다.

기존 코드에는 다음 설정이 포함되어 있습니다.

```python
RUN_MATRIX = True
RUN_YOLO_SCALES = True
```

전체 노트북을 실행하면 모델 비교와 YOLO 추가 학습이 시작될 수 있으므로 필요한 실험을 선택해야 합니다.

Faster R-CNN의 출력 폴더와 YOLO의 실행 이름을 중복 사용하면 기존 결과가 덮어써질 수 있습니다. 새 실험에서는 고유한 실행 이름과 저장 경로를 사용하는 것이 좋습니다.

또한 `MATRIX_LR` 변수에 0.0005가 지정되어 있지만 실제 `train_one` 함수의 Optimizer에서는 Learning Rate 0.005가 고정되어 있습니다. 학습 조건을 재현할 때 실제 Optimizer에 전달되는 값을 확인해야 합니다.

## 5. 확장 파이프라인

### 5.1 Segmentation 라벨 변환

**노트북 및 스크립트**
- `03.6_coco_to_yolo_seg.ipynb`
- `06A_convert_coco_to_yolo_seg.py`

COCO Polygon annotation을 YOLO Segmentation 형식으로 변환하는 코드를 작성했습니다.

스크립트 옵션은 다음 명령어로 확인할 수 있습니다.

```bash
python scripts/06A_convert_coco_to_yolo_seg.py --help
```

주요 기능은 Category 변환, 좌표 정규화, 잘못된 Polygon 필터링 및 라벨 저장입니다.

`--dry-run`을 사용해도 디렉터리 생성이 발생할 수 있으므로 출력 경로를 확인해야 합니다.

이 스크립트는 라벨을 생성하는 기능이며, Segmentation 모델의 학습과 평가는 별도로 진행해야 합니다.

### 5.2 ROI Crop 파이프라인

**스크립트:** `06B_build_roi_crop_pipeline.py`

정답 병변 bbox를 기준으로 ROI를 추출하고 이미지 및 Polygon 좌표를 변환하는 파이프라인입니다.

`--yolo-seg-dir`과 `--dry-run` 옵션을 지원합니다.

이 방식은 정답 위치를 사용하는 학습 데이터 준비 단계이므로 실제 추론에서는 별도의 ROI 탐지 과정이 필요합니다.

### 5.3 2단계 ROI 재탐지

**스크립트:** `06C_roi_crop_cascade.py`

1차 탐지 결과에서 ROI를 추출하고 2차 모델로 재탐지하는 구조입니다.

```text
1차 탐지
    ↓
예측 bbox 그룹화
    ↓
ROI Crop
    ↓
2차 탐지
    ↓
원본 좌표 복원
    ↓
NMS
```

현재 스크립트는 다른 코드에서 함수를 불러 사용하는 Helper 모듈이며 독립 실행 Entry Point는 없습니다.

실제 두 모델을 이용한 학습·추론 성능 비교는 완료하지 않았습니다.

또한 현재 TP/FP/FN 계산 함수는 예측 점수 기반의 공식 AP 계산과 다르므로 AP 평가 결과로 해석해서는 안 됩니다.

## 6. 실행 과정에서 확인한 문제

### 데이터 경로 및 실행 순서

`05D_backbone_experiment_matrix.ipynb`에는 `random` 모듈을 불러오지 않아 발생한 `NameError`가 있습니다.

`05E_compare_all_results.ipynb`에서는 `load_frcnn_rows` 함수 정의 셀과 실행 순서가 맞지 않아 오류가 발생했습니다.

두 노트북 모두 전체 재실행에 앞서 셀 의존 관계와 실행 순서를 확인해야 합니다.

또한 `05D`에서 CLAHE+Gamma 폴더를 `02_clahe_gamma`로 참조하지만, 실제 전처리 경로는 `04_clahe_gamma`입니다.

### 모델 구성 관련 주의사항

`05C`의 ConvNeXt 및 Swin 관련 구현은 실제 외부 Backbone이 아니라 인터페이스 검증용 스켈레톤입니다.

특히 ConvNeXtTinyStub의 `out_channels` 선언과 실제 특징 맵 채널 수가 일치하지 않는 부분이 있어 실제 탐지 모델에 연결하기 전에 수정해야 합니다.

### 결과 집계 및 저장 경로

`05E`의 `groupby.first()`는 열별 첫 번째 결측이 아닌 값을 선택하므로 결측값이 있을 경우 서로 다른 행의 값이 하나의 결과로 결합될 가능성이 있습니다.

또한 여러 학습 실행이 고정된 출력 폴더를 사용하면 이전 실험 결과를 덮어쓸 수 있습니다.

새로운 실험에서는 실행 이름과 출력 폴더를 분리하고, 평가 기준과 체크포인트를 함께 관리해야 합니다.

### 추가 확인이 필요한 사항

- `05B`와 `05D`의 Negative Image 포함 방식 차이
- Faster R-CNN과 YOLO 간 평가 방식 차이
- 원본 학습 CSV 및 체크포인트 보존 여부
- `torch.load(..., weights_only=False)`를 사용할 경우 가중치 파일의 신뢰성
- ROI Cascade의 TP/FP/FN과 정식 AP 평가 방식 차이
- 저장된 결과와 노트북 전체 재실행 결과의 일치 여부

## 7. 재현 범위와 한계

현재 저장소에는 실험 코드, 노트북, 집계 결과와 주요 시각화 자료를 정리했습니다.

다만 원본 ARCADE 데이터, 학습 가중치, 전체 학습 CSV 및 정확한 패키지 Lockfile은 포함되어 있지 않습니다.

따라서 노트북의 기존 결과를 열람하거나 코드 구조를 살펴볼 수 있지만, 동일한 환경에서 모든 모델을 다시 학습해 동일한 수치를 얻을 수 있는지는 별도 검증이 필요합니다.

향후 재현성을 높이기 위해서는 다음 작업이 필요합니다.

- 실제 설치 패키지와 버전 고정
- 데이터셋 버전 및 파일 Checksum 확인
- 데이터 분할과 라벨 변환 과정 검증
- 실험별 독립 출력 폴더 구성
- 학습 설정 및 Checkpoint 선택 기준 기록
- 전체 노트북의 순차 실행 검증
- 별도 Test 데이터셋과 반복 실험을 통한 평가

원본 데이터의 공개 범위와 라이선스 관련 사항은 [공개 범위 검토 문서](publication_review.md)에 정리했습니다.

실험의 세부 과정과 결과는 [실험일지](experiment_summary.md) 및 [결과 설명](../results/README.md)을 참고할 수 있습니다.
