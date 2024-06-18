import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn
import numpy as np

# 定义图像预处理转换
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的ResNet50模型
model = resnet50(pretrained=True)
model.eval()  # 设置为评估模式

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

with torch.no_grad():
    while True:
        # 捕捉视频帧
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，退出")
            break

        # 转换为RGB格式
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 应用预处理
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

        # 进行预测
        outputs = model(img_tensor)

        # 获取预测结果
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

        # 在视频帧上显示预测结果
        cv2.putText(frame, f'Predicted Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # 显示结果帧
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) == ord('q'):
            break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
