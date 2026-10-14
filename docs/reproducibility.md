# 실행 환경과 재현성

## 환경 근거

현재 정리 작업의 Python 3.12.14와 아래 **과거 실험 환경**은 구분합니다. 원본 노트북 metadata와 저장된 출력에 나타난 값만 기록합니다.

| 구성 | 확인된 기록 | 출처·제약 |
|---|---|---|
| Python | 3.12.6 | 10개 notebook metadata 및 YOLO 런타임 로그. `03.6`은 version metadata 없음 |
| Ultralytics | 8.4.110 | 04 cell7, 05B cell11, 05C cell8 저장된 로그. 현재 배포/설치 가능 여부 미검증 |
| PyTorch | 2.13.0+cu130 | 05B cell2 및 YOLO 로그. 빌드 출처/현재 배포 여부 미검증 |
| CUDA runtime | 13.0 | 05B cell2. 드라이버 버전은 확인 필요 |
| GPU | NVIDIA GeForce RTX 5090, 32111MiB | 저장된 YOLO 로그. 현재 작업 환경에 동일 GPU가 제공된다는 뜻이 아님 |
| scikit-image | 0.26.0 | 02 cell15 출력 |
| seaborn | 0.13.2 | 03_quality cell6 출력 |
| torchvision/OpenCV/numpy/pandas/Pillow/matplotlib/tqdm/PyYAML | 확인 필요 | import는 있으나 버전 출력 없음. `01` OpenCV 버전 셀은 미실행 |
| JupyterLab/ipykernel/nbformat | 확인 필요 | 노트북 사용/검증 도구. 과거 버전 lockfile 없음 |

`requirements.txt`는 import 기반 목록입니다. 정확한 설치 조합·전이 의존성이 완전히 확인되지 않아 **핀을 추측하지 않았습니다**. 확인된 일부 버전도 전체 호환 환경을 검증하지 않았으므로 주석과 이 문서에 근거만 기록합니다. MMDetection/MMCV/timm/pycocotools는 현재 원본 구현의 필수 import가 아니므로 포함하지 않았습니다.

과거 서버에서 사용자 확인 후 필요한 정보: `python --version`, `python -m pip freeze`, `torch/torchvision` 버전, CUDA/드라이버, GPU 모델, Ultralytics 실제 설치 출처, 실행별 `args.yaml`/CSV. 출력에 인증정보나 사설 주소가 없는지 검수한 뒤 공유합니다. 환경 복원을 위해 비밀정보를 채팅이나 requirements에 넣지 않습니다.

## 디렉터리와 실행 순서

모든 노트북은 저장소의 `notebooks/`를 cwd로 사용합니다. 상대경로 변경은 이 전제를 사용합니다. `data/`는 공개하지 않는 외부 입력이며 데이터 ZIP 자동 다운로드 주소를 추측해 추가하지 않았습니다.

```text
data/arcade/arcade/stenosis/
  train/images/*.png
  train/annotations/train.json
  val/images/*.png
  val/annotations/val.json
data/preprocessed/<pipeline>/{train,val}/*.png
data/yolo/<pipeline>/images/{train,val}/*.png
data/yolo/<pipeline>/labels/{train,val}/*.txt
```

1. 데이터 사용 권한·버전·환자별 분할을 확인하고 별도 저장합니다.
2. 새 가상환경에서 CPU/CUDA에 맞는 공식 torch/torchvision 조합을 선택합니다. 그 후 `requirements.txt`를 설치하고 설치 버전을 기록합니다. 이전 수치 그대로 재현하는 lockfile은 아직 없습니다.
3. `01` 데이터 분석, `02` 전처리, `03_quality` 품질평가를 순서대로 실행합니다. 랜덤 예시 선택은 seed가 고정되어 있지 않습니다. 한 셀만 수정/실행한 기록이 있으므로 출력이 현재 소스 전체 실행과 일치한다고 가정하지 않습니다.
4. `03.5`에서 images/labels 모두 준비하고 `04` 데이터 개수 check를 통과한 뒤 학습합니다. `04`는 images/train와 images/val 동기화를 제공하며 유효한 labels는 먼저 준비해야 합니다.
5. `04` 기본값: YOLO11n pretrained, epochs100,imgsz640,batch16,device0,workers8,patience30,seed42. CPU를 쓰면 device를 `'cpu'`로 바꾸어야 하며 이 변경은 과거 환경과 다릅니다.
6. `05A`: 새 실험이면 `use_manual=False`로 변경합니다. 그대로 두면 원래의 수동 수치가 새 결과를 덮어씁니다.
7. `05B`: 실행할 matrix/YOLO scale을 명시적으로 선택합니다. 원본은 `RUN_MATRIX=True`, `RUN_YOLO_SCALES=True`이고 단일 실행은 100 epoch입니다. 결과가 같은 폴더에 저장되므로 새 run 이름/디렉터리를 먼저 정해야 합니다.
8. `03.6`/06A segmentation 및 06B/06C 확장은 실제 결과가 없는 별도 개발 단계입니다. 06A는 labels만 생성하므로 images 구성·data.yaml·학습·평가를 추가로 준비해야 합니다.

06A의 CLI `--data-dir`, `--out-root`를 지정할 수 있습니다. `--dry-run`은 원본 구현상 label_dir.mkdir를 먼저 호출하므로 디렉터리는 생성될 수 있습니다. 06B는 `--yolo-seg-dir`, `--dry-run` 옵션이 있고 06C는 실행 entrypoint가 없는 helper 모듈입니다. 외부 입력 경로와 출력 경로를 분리하여 원본 데이터를 덮어쓰지 않습니다.

## 기록된 실패와 수정되지 않은 문제

| 문제 | 증거와 다음 확인 | 정리본에서의 처리 |
|---|---|---|
| 전처리 val images=0 | 03.5 cell10 CHECK; 04 cell4에서 두 선택 파이프라인 val200 복구 | 코드 알고리즘 보존, 양쪽 기록 유지 |
| 05D `random` NameError | cell2 import를 실행하지 않고 cell4 실행한 기록 | 오류 보존. 수정 코드로 성공한 것처럼 표시하지 않음 |
| 05E `load_frcnn_rows` NameError | 정의 cell3와 실행 cell4 순서 문제. 이후 표는 다른 실행 상태일 수 있음 | 오류와 후속 결과 둘 다 보존 |
| 잘못된 05D CG 경로 | `02_clahe_gamma` vs 실제 `04_clahe_gamma` | 수정하지 않고 별도 후속 변경 필요로 보고 |
| 05B matrix LR 설정 미반영 | 상단 MATRIX_LR=0.0005지만 train_one optimizer는 lr=0.005 고정 | 실행 조건 설명에 실제 optimizer 값을 사용, 알고리즘 변경 없음 |
| 05B/05D loader 차이 | 05B는 negative 유지, 05D는 annotation 있는 이미지만 순회 | 05D 결과 미실행, 같은 조건 비교로 단정하지 않음 |
| 05C stub channel/stride 가정 | ConvNeXtTinyStub out_channels 선언 [192,384,768]와 실제 출력 [384,768,768] 불일치, Swin stub 실제 stride16/32/32 | Swin toy 기록만 확인. 모든 custom detector가 완성됐다고 표현하지 않음 |
| 평가기 차이 | FRCNN 11-point AP50 vs Ultralytics mAP; 최고 epoch 선택 기준도 다름 | 공통 evaluator 도입은 후속 과제 |
| 05E `groupby.first()` | 열별 first non-null을 취하므로 누락 시 서로 다른 행의 값이 섞일 수 있음 | 원본 집계 표 보존하되 원본 CSV 대조 필요 |
| 반복 실행 덮어쓰기 | 고정 FRCNN output 폴더, YOLO `exist_ok=True` | 저장된 history를 셀별 파일로 분리. 새 run에는 고유 이름 필요 |
| 신뢰할 수 없는 weight 로딩 | 05E `torch.load(..., weights_only=False)` | 원본 코드 유지. 출처가 확인된 가중치만 사용하고 외부 weight는 검증 필요 |
| ROI 평가 helper | 06C match helper는 scores를 받지 않고 입력 순서대로 matching | TP/FP/FN을 공식 AP로 표현하지 않음. cascade 실제 평가 필요 |
| seaborn FutureWarning | 03_quality palette without hue | 반복 warning 출력만 요약, plotting 코드는 유지 |

## 이번에 바뀐 실행 관련 코드

- 개인 서버의 절대경로를 notebook cwd의 부모 디렉터리로 변경했습니다. 06A는 스크립트 위치 기준 프로젝트 `data/`를 기본값으로 사용하며 기존 CLI override를 유지합니다.
- `03.6` cell7은 converter 파일이 존재하면 쓰지 않습니다. 원본 notebook 내 embedded converter는 독립 06A의 축약 버전이므로 기존 완전한 스크립트 덮어쓰기를 막았습니다.
- 모델 구조, loss, optimizer, seed, 학습 epoch, threshold, 라벨 변환·crop·평가 알고리즘은 수정하지 않았습니다. 셀 이름·순서·execution_count는 유지했습니다.
- `03.6`의 원본 13개 셀에 누락된 notebook cell ID를 추가했습니다. 형식 검증 경고를 해소하는 metadata 변경이며 source/실행 결과를 추가하지 않습니다.
- 과거 경로 출력은 `<PROJECT_ROOT>`로 익명화했습니다. 이는 학습 산출물의 원래 저장 위치를 새 위치에서 검증했다는 뜻이 아닙니다.

각 변경 전/후 코드와 hash는 `code_changes.json`에 있습니다. 변경 전 개인 경로는 `<ORIGINAL_PRIVATE_ROOT>`로 가렸으며 `before_sha256`은 실제 원본 코드 기준입니다. 완전한 원본은 공개 저장소 밖에 보존했습니다.
