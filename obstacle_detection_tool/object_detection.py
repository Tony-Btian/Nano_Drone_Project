import numpy as np
import cv2
import os

# 相机内参（根据相机校准结果获取）
fx = 500  # 焦距x
fy = 500  # 焦距y
cx = 320  # 光心x
cy = 240  # 光心y

# 深度图文件路径
depth_map_path = 'depth_image.png'

# 检查文件是否存在
if not os.path.exists(depth_map_path):
    raise FileNotFoundError(f"深度图文件 '{depth_map_path}' 不存在，请检查文件路径和文件完整性。")

# 读取深度图
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

# 检查深度图是否成功读取
if depth_map is None:
    raise ValueError(f"无法读取深度图文件 '{depth_map_path}'，请检查文件格式和内容。")

# 检查图像通道数
if len(depth_map.shape) == 3:
    # 如果是多通道图像，提取第一个通道作为深度图
    depth_map = depth_map[:, :, 0]

# 获取图像尺寸
height, width = depth_map.shape

# 创建网格（像素坐标）
u, v = np.meshgrid(np.arange(width), np.arange(height))

# 提取深度值
d = depth_map.astype(np.float32)

# 计算三维空间坐标
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d

# 将结果转换为点云格式（N, 3）
points_3d = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

# 打印一些调试信息
print("Depth map shape:", depth_map.shape)
print("Min depth:", np.min(d))
print("Max depth:", np.max(d))

# 阈值法检测障碍物区域
threshold = 15.0  # 设定深度阈值（根据需要调整）
obstacle_mask = d < threshold

# 提取障碍物区域的三维点
obstacle_points = points_3d[obstacle_mask.flatten()]

# 检查是否检测到障碍物
if obstacle_points.size == 0:
    raise ValueError("No obstacle points detected. Try adjusting the threshold.")

# 计算质心
centroid = np.mean(obstacle_points, axis=0)
print(f"Obstacle Centroid: {centroid}")

# 计算最小包围矩形（仅考虑X-Y平面）
min_x, min_y = np.min(obstacle_points[:, :2], axis=0)
max_x, max_y = np.max(obstacle_points[:, :2], axis=0)
bounding_box = [(min_x, min_y), (max_x, max_y)]
print(f"Obstacle Bounding Box: {bounding_box}")

# 可视化质心和最小包围矩形
import matplotlib.pyplot as plt

plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], c='b', marker='o', s=1)
plt.scatter(centroid[0], centroid[1], c='r', marker='x')  # 质心
plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'g-')  # 最小包围矩形
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Obstacle Detection')
plt.show()
