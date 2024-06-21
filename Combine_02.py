import cv2
import numpy as np
import torch

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 验证CUDA可用性
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Switching to CPU.")


# 加载YOLOv5模型
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device, force_reload=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', force_reload=True)

# 加载Midas深度估计模型
try:
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True).to(device)
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")


def detect_objects(frame):
    # 使用模型进行目标检测
    results = model(frame)
    return results.pandas().xyxy[0]  # 返回检测结果


def estimate_depth(frame):
    # 将图像转换为Midas模型的输入格式
    transform = midas_transforms.dpt_transform
    input_batch = transform(frame).to(device)
    
    # 进行深度估计
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    return depth_map


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:

        # 读取摄像头画面
        frames = []
        for _ in range(2):
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面")
                return
            frames.append(frame)

        # 获取画面的高度和宽度
        height, width = frames[0].shape[:2]

        # 目标检测
        results = detect_objects(frames[0])
        
        # 在原始图像上绘制检测结果
        for idx, row in results.iterrows():
            x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            label = f"{cls} {conf:.2f}"
            # 绘制边框和标签
            cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frames[0], label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 生成深度图
        depth_map = estimate_depth(frames[1])

        # 创建双视图图像
        combined_frame = cv2.hconcat([frames[0], depth_map])
        
        # 显示画面
        cv2.imshow('Camera View', combined_frame)
        
        # 检查是否按下'q'键，按下则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
