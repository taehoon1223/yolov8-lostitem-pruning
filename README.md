#  YOLOv8 기반 실시간 객체 탐지 및 모델 경량화 프로젝트

본 프로젝트는 Ultralytics YOLOv8을 기반으로 **객체 탐지 모델 학습**, **이미지 테스트**, **실시간 성능(FPS) 측정**,  
그리고 **Torch-Pruning 기반 모델 경량화 및 시각적 성능 평가**를 수행합니다.

---

##  프로젝트 구성

```bash
codes/
├── yolov8_basic.py         # 기본 YOLOv8 학습 스크립트
├── yolov8_pruning.py       # 모델 Pruning + mAP & MACs 시각화
├── image_test.py           # 이미지 추론 및 결과 저장
├── fps_test.py             # FPS 측정 스크립트 (배치 지원)
├── weights/                # 학습된 모델 저장 폴더
├── dataset/                # 학습용 이미지 및 data.yaml
├── configs/                # pruning_config.yaml (선택)
├── results/                # 성능 변화 그래프 등 출력 결과

