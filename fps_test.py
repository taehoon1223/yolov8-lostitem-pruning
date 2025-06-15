import time
import os
import cv2
import torch
from ultralytics import YOLO
import torch.nn as nn
import sys

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

def measure_fps_realistic_with_batch(
    model_path, image_dir, input_size=(640, 640), num_images=100, batch_size=4, device='cuda'
):
    """
    배치 크기를 지원하는 FPS 측정 함수.

    :param model_path: 학습된 모델의 경로 (best.pt)
    :param image_dir: 테스트할 이미지들이 저장된 디렉터리 경로
    :param input_size: 입력 이미지 크기 (기본값: 640x640)
    :param num_images: FPS 측정을 위해 처리할 이미지 수
    :param batch_size: 한 번에 처리할 이미지 수 (배치 크기)
    :param device: 사용할 디바이스 ('cuda' 또는 'cpu')
    :return: FPS (Frames Per Second)
    """
    # 모델 로드
    model = YOLO(model_path)
    model.to(device)  # 디바이스 설정

    # 이미지 파일 리스트 가져오기
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        raise ValueError("이미지 디렉터리에 사용할 수 있는 이미지가 없습니다.")
    
    # 테스트할 이미지 수 제한
    image_files = image_files[:num_images]

    # 이미지 데이터 로드 및 전처리
    images = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.resize(img, input_size)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
    images = torch.stack(images).to(device)  # 이미지를 텐서로 스택

    # Warm-up 단계 (배치 처리)
    for _ in range(5):
        batch = images[:batch_size]  # 첫 배치만 사용
        _ = model(batch)

    # FPS 측정을 위한 타이머 시작
    start_time = time.time()
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]  # 배치 생성
        _ = model(batch)
    end_time = time.time()

    # FPS 계산
    total_time = end_time - start_time
    fps = num_images / total_time

    print(f"Processed {num_images} images in {total_time:.2f} seconds with batch size {batch_size}")
    print(f"FPS (realistic with batch size {batch_size}): {fps:.2f}")
    return fps


if __name__ == "__main__":
    model_path = ""  # 모델 경로
    image_dir = ""  # 테스트할 이미지가 저장된 디렉터리 경로

    batch_size = 1  # 배치 크기 설정
    fps = measure_fps_realistic_with_batch(
        model_path, image_dir, input_size=(640, 640), num_images=100, batch_size=batch_size, device='cuda'
    )
