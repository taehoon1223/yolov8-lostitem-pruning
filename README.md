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

행 방법
학습
python yolov8_basic.py

추론
python image_test.py

FPS 측정
python fps_test.py

모델 Pruning 및 Fine-tuning
python yolov8_pruning.py --model weights/best.pt --cfg configs/pruning_config.yaml

| 항목           | 내용                                   |
| ------------ | ------------------------------------ |
| 모델           | YOLOv8s (Ultralytics)                |
| 학습 이미지 크기    | 640x640                              |
| 학습 Epoch     | 160                                  |
| Pruning 알고리즘 | Torch-Pruning (GroupNorm 기반)         |
| 시각화 그래프      | results/pruning\_perf\_change.png 포함 |


주요 기술 스택
Ultralytics YOLOv8

Torch-Pruning

PyTorch

OpenCV

NumPy

Matplotlib

참고 사항
fps_test.py는 배치 크기를 조절하여 실사용 기준 FPS를 측정합니다.

yolov8_pruning.py는 iterative pruning + fine-tuning 과정을 반복하며 성능을 기록합니다.

Pruning 이후 mAP 감소가 --max-map-drop 이상이면 자동 중단됩니다.

pruning_perf_change.png는 pruning 단계별 mAP 및 MACs 변화를 시각화한 결과입니다.


