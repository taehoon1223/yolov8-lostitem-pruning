from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck

# 사용자 정의 레이어 C2f_v2 클래스 정의
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


# 모델 로드
model = YOLO('C:')

# 테스트할 이미지 경로 리스트
images = ['C:/', 
          'C:/']

# 예측 실행
results = model.predict(source=images, conf=0.25, save=True)

# 결과 출력
for idx, result in enumerate(results):
    print(f"Result for image {idx + 1}:")
    print(result)  # 예측 결과 요약 정보 출력
    
    # 개별 이미지 결과 시각화
    result.plot(save=True)
