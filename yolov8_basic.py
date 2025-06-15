from ultralytics import YOLO

def train_yolo():
    # YOLOv8 모델 불러오기
    model = YOLO('yolov8s.pt')  # 사전학습된 YOLOv8 모델
    
    # 데이터 경로
    data_path = "C:/"  # 데이터셋 yaml 파일 경로
    
    # 모델 학습
    model.train(
        data=data_path,        # 데이터셋 yaml 파일 경로
        epochs=160,             # 학습 에포크 수
        batch=16,              # 배치 크기
        imgsz=640,             # 이미지 크기
        save_period=10,        # 몇 에포크마다 가중치 저장
        project="yolo_basic",  # 저장 프로젝트 이름
        name="experiment1",    # 실험 이름
        workers=4,             # 데이터 로드 워커 수
        device=0               # 사용할 GPU 번호
    )

if __name__ == '__main__':
    train_yolo()
