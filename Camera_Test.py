import cv2
import numpy as np

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取画面")
            break
        
        # 获取画面的高度和宽度
        height, width = frame.shape[:2]

        # depth_map_color = cv2.applyColorMap()
        # 创建一个新图像，宽度为原始画面的两倍
        dual_frame = cv2.hconcat([frame, frame])
        
        # 显示画面
        cv2.imshow('Dual View', dual_frame)
        
        # 检查是否按下'q'键，按下则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
