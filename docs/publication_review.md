# 공개 위험 검수와 원본 보존

## 보존 위치와 공개 범위

원본 `yolo.zip`은 수정하지 않았습니다. SHA-256은 `source_manifest.json`에 기록했습니다. ZIP에는 ipynb 11개와 py 3개만 있으며 별도 dataset ZIP, raw medical image 파일, COCO JSON, 학습 weights, 원본 result CSV는 없습니다. **노트북 출력 안에는 의료영상이 포함되어 있습니다.**

로컬 원본·전체 출력·이미지는 `/workspace/xca-yolo-review/` 및 업로드 첨부 경로에 보존했으며 Git 저장소 밖입니다. 원본 ZIP과 비공개 이미지 폴더를 공개 포트폴리오 ZIP/PR에 포함하지 않습니다. 해당 로컬 보존은 외부 영구 저장소 백업을 의미하지 않으므로 원본 업로드 파일도 별도로 보관해야 합니다.

| 검사 항목 | 발견 | 처리·확인 사항 |
|---|---|---|
| 의료영상·crop | 01의 예시 5개, 02의 전처리/튜닝 시각화 12개 | 17개 PNG를 그대로 디코딩하여 SHA-256 검증 후 로컬 비공개 보존. 공개 notebook image 출력은 제외 |
| 집계 그래프 | bbox 분포, 품질, YOLO 비교, 종합 비교 총5개 | 의료영상 픽셀 없는 집계 그래프로 육안 검토. 원본 PNG를 공개 후보 results에 추출 |
| 원본 annotation | 01 cell19 첫 COCO annotation의 polygon/bbox, 02 cell7/8 개별 이미지 ID/bbox | 전체 output은 로컬 보존. 개별 원본 annotation 출력은 공개 notebook/evidence에서 제외. 합계 통계 유지 |
| 개인 절대경로 | 코드와 로그의 개인 서버 home path | 코드 상대경로화; 공개 로그는 `<PROJECT_ROOT>`로 익명화. 변경 전 코드 문서의 개인경로도 가림 |
| 인증정보 | GitHub token/AWS key/private key/email/credential assignment 패턴 검사 | 일치 없음. 셀 source·텍스트/HTML·metadata를 검토. 알려진 패턴 부재는 모든 비밀정보 부재의 절대 보증이 아님 |
| 대용량 모델·원본 데이터 파일 | ZIP 독립 파일로는 없음 | weights/data/archive/annotation 경로 ignore 강화. force-add나 새 notebook 출력은 별도 검수 필요 |
| 환자 식별정보 | 이미지의 명시적 식별정보 유무·비식별화 절차를 입증할 기록 없음 | 이미지 전부를 공개하지 않음. 데이터 권한·비식별화 근거·라이선스 확인 필요 |
| notebooks widget metadata | progress widget 출력·widget state | 원본 로컬 보존 후 공개 복사본의 progress widget/state 정리 |
| 코드/모델 라이선스 | 별도 원본 LICENSE 자료 없음 | 사용 라이브러리·사전학습 모델·데이터의 라이선스 확인 필요. 임의 라이선스 파일 추가하지 않음 |

의료영상이 처리된 비교 grid도 원본 데이터에서 파생되므로 공개 권한을 확인하기 전에는 공개하지 않습니다. 결과 manifest의 공개 제외 표시와 hash만으로 의료영상 자체를 복원할 수는 없습니다.

## 출력 정리의 보존 절차

1. 원본 ZIP CRC 검증과 SHA-256 기록 후 안전한 별도 폴더에 풀었습니다. 경로 traversal/symlink entry는 허용하지 않았습니다.
2. 원본 파일 14개 모두 별도 보존, byte-identical 중복 파일이 없는지 비교했습니다.
3. 모든 원본 code cell output 객체를 notebook별 JSON에 보존하고 다시 읽어 구조적으로 동일함을 확인했습니다.
4. 원본 PNG 22개를 디코딩하여 파일로 추출, 읽기·해시 확인했습니다. 중요한 수치 표를 HTML/CSV로, FRCNN 190개 epoch 행을 셀별 CSV로 추출했습니다.
5. 공개 정리본에서 progress/warning/폴더 나열을 정리하고 최종 검증/epoch 숫자/오류·유의미한 표를 유지했습니다. 의료영상과 개별 원본 annotation 출력은 공개 범위 때문에 따로 제외했습니다.
6. 원본 ZIP, 원본 파일, PNG, 전체 output JSON의 보존 여부를 검증한 뒤 최종 검토 자료를 만들었습니다. 소스 셀 삭제는 없습니다.

## 공개 전 확인 필요

- ARCADE 공식 출처, 데이터 버전·사용/파생 결과 공개 조건 및 환자 단위 split 근거.
- 사용 코드·사전학습 가중치의 라이선스와 공개 조건.
- 원래 환경 정보와 학습 CSV/args.yaml/체크포인트의 확보 가능 여부.
- GitHub push에 대한 사용자 승인. `main` 직접 반영·병합 없이 별도 브랜치/PR에서 검토합니다.

기존 저장소 파일을 삭제하지 않았습니다. 원본 ZIP/추출 파일/의료영상도 삭제하지 않았습니다. 공개 복사본의 출력 정리는 원본을 보존한 상태에서 이루어졌습니다.
