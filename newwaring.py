import time
import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import winsound  # 경고음을 위한 모듈 (Windows용)
import sys  # sys 모듈 추가
import threading  # 비동기 경고음 실행을 위한 스레드

# 사용자 정의 Conv 클래스
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 사용자 정의 Bottleneck 클래스
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):  
        super().__init__()
        self.conv1 = Conv(c1, int(c2 * e), k[0], 1, g)
        self.conv2 = Conv(int(c2 * e), c2, k[1], 1, g)
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

# 사용자 정의 C2f_v2 클래스
class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  
        super().__init__()
        self.c = int(c2 * e)  
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# 사용자 정의 클래스 등록
sys.modules['C2f_v2'] = C2f_v2

# 비동기 경고음 함수
alert_event = threading.Event()

def play_alert_sound_limited():
    if not alert_event.is_set():
        alert_event.set()
        winsound.Beep(1000, 70)  # 주파수 1000 Hz, 100ms 경고음
        time.sleep(0.1)
        alert_event.clear()

# 클래스 ID와 이름 매핑
CLASS_NAMES = ['bag', 'cap', 'card', 'earphone', 'glasses', 'person', 'phone', 'umbrella', 'wallet']
ALERT_CLASSES = [0, 1, 2, 3, 4, 6, 7, 8]  # 경고음을 출력할 클래스 ID

def detect_video_with_alert(model_path, video_path, output_path, device=None):
    # 디바이스 설정
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = YOLO(model_path)
    model.to(device)

    # 동영상 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"동영상 파일 {video_path}를 열 수 없습니다.")

    # 출력 동영상 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int(400 / fps)  # FPS에 따른 대기 시간 계산
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),  # MP4 코덱
        fps,
        (frame_width, frame_height)
    )

    # 동영상 처리
    frame_count = 0
    total_time = 0
    detection_started = False
    detection_start_frame = fps*2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1초 후 검출 시작
        if frame_count < detection_start_frame:
            out.write(frame)
            cv2.putText(frame, "Waiting for detection to start...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("YOLOv8 Detection with Alert", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break
            continue

        if not detection_started:
            print("Object detection started at frame:", frame_count)
            detection_started = True

        # YOLO 모델 실행
        results = model(frame)

        # 감지 결과 가져오기
        detected_objects = results[0].boxes
        alert_triggered = False
        detected_names = []  # 감지된 객체 이름 저장

        for obj in detected_objects:
            cls = int(obj.cls.item())
            conf = obj.conf.item()
            if cls != 5 and conf > 0.5:  # 'person'(ID: 5)을 제외
                detected_names.append(CLASS_NAMES[cls])  # 클래스 이름 추가
                if cls in ALERT_CLASSES:
                    alert_triggered = True

        # 화면에 클래스 이름 출력
        if detected_names:
            cv2.putText(
                frame,
                f"Detected: {', '.join(detected_names)}",  # 클래스 이름 출력
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # 경고음 출력
        if alert_triggered:
            cv2.putText(frame, "Warning: Object Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            threading.Thread(target=play_alert_sound_limited).start()

        # 결과 그리기
        annotated_frame = results[0].plot()

        # 출력 동영상에 저장
        out.write(annotated_frame)

        # 실시간 보기
        cv2.imshow("YOLOv8 Detection with Alert", annotated_frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    # 리소스 정리
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:/Users/ghks5//ultralytics/model_in_person/new0.5/best.pt"
    video_path = "C:/Users/ghks5/ultralytics/mp4/in/4_.mp4"
    output_path = "C:/Users/ghks5/ultralytics/mp4/out/4.mp4"

    # 동영상 처리 및 경고 기능 실행
    detect_video_with_alert(model_path, video_path, output_path, device='cuda')
