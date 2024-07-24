import cv2
import numpy as np

# 回调函数，用于滑块操作
def update_threshold(val):
    global distance_threshold
    distance_threshold = val
    process_image()

# 处理图像并更新显示
def process_image():
    global depth_image, distance_threshold

    # 创建一个与深度图相同大小的掩码图像，初始化为零
    obstacle_mask = np.zeros_like(depth_image, dtype=np.uint8)

    # 将深度值小于阈值的区域设置为255（白色），表示障碍物
    obstacle_mask[depth_image < distance_threshold] = 255

    # 使用形态学操作来处理噪声
    kernel = np.ones((5, 5), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

    # 找到障碍物轮廓
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始深度图上绘制障碍物轮廓
    depth_image_with_contours = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(depth_image_with_contours, contours, -1, (0, 0, 255), 2)

    # 显示原始深度图
    cv2.imshow('Original Depth Image', depth_image)

    # 显示识别出的障碍物
    cv2.imshow('Obstacle Mask', obstacle_mask)

    # 显示带有障碍物轮廓的深度图
    cv2.imshow('Depth Image with Obstacles', depth_image_with_contours)

# 读取深度图
depth_image = cv2.imread('depth_image.png', cv2.IMREAD_UNCHANGED)

# 确保深度图是单通道
if len(depth_image.shape) == 3:
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

# 初始化距离阈值
distance_threshold = 2000

# 创建窗口
cv2.namedWindow('Original Depth Image')
cv2.namedWindow('Obstacle Mask')
cv2.namedWindow('Depth Image with Obstacles')

# 创建滑块
cv2.createTrackbar('Distance Threshold', 'Depth Image with Obstacles', distance_threshold, 5000, update_threshold)

# 初始处理图像
process_image()

# 等待用户输入
cv2.waitKey(0)
cv2.destroyAllWindows()
