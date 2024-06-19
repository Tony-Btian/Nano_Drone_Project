import cv2
import numpy as np
import torch

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(frame):
    # 使用模型进行目标检测
    results = model(frame)
    return results.pandas().xyxy[0]  # 返回检测结果

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:
        
        # 读取摄像头画面
        frames = []
        for _ in range(4):
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面")
                return
            frames.append(frame)
        
        # 获取画面的高度和宽度
        height, width = frames[0].shape[:2]

        # 目标检测
        results = detect_objects(frame[0])
        
        # 在原始图像上绘制检测结果
        for idx, row in results.iterrows():
            x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            label = f"{cls} {conf:.2f}"
            # 绘制边框和标签
            cv2.rectangle(frames[1], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frames[1], label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 创建四宫格图像
        top_row = cv2.hconcat([frames[0], frames[1]])
        bottom_row = cv2.hconcat([frames[2], frames[3]])
        quad_fram = cv2.vconcat([top_row, bottom_row])
        
        # 显示画面
        cv2.imshow('Quad View', quad_fram)
        
        # 检查是否按下'q'键，按下则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
