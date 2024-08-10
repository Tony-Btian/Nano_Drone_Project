import numpy as np
import cv2
import os
import pyvista as pv
from sklearn.cluster import DBSCAN

# 相机内参（根据相机校准结果获取）
fx = 500  # 焦距x
fy = 500  # 焦距y
cx = 320  # 光心x
cy = 240  # 光心y

# 深度图文件路径
depth_map_path = r'E:\Anton\Documents\GitHub\MSc_Project\obstacle_detection_tool\depth_map.png'

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

# 将深度图归一化到0-255范围
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 使用Canny边缘检测
edges = cv2.Canny(depth_map_normalized, 50, 150)

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

# 提取障碍物区域的三维点
obstacle_mask = edges > 0
obstacle_points = points_3d[obstacle_mask.flatten()]

# 检查是否检测到障碍物
if obstacle_points.size == 0:
    raise ValueError("No obstacle points detected. Try adjusting the threshold or detection method.")

# 使用DBSCAN进行聚类
clustering = DBSCAN(eps=5, min_samples=10).fit(obstacle_points)
labels = clustering.labels_

# 获取不同障碍物的索引
unique_labels = set(labels)
obstacle_centroids = []
bounding_boxes = []

for label in unique_labels:
    if label == -1:
        continue  # 忽略噪声点
    label_mask = (labels == label)
    points = obstacle_points[label_mask]
    
    # 计算质心
    centroid = np.mean(points, axis=0)
    obstacle_centroids.append(centroid)
    
    # 计算最小包围矩形（仅考虑X-Y平面）
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)
    bounding_box = [(min_x, min_y, centroid[2]), (max_x, min_y, centroid[2]), 
                    (max_x, max_y, centroid[2]), (min_x, max_y, centroid[2])]
    bounding_boxes.append(bounding_box)

# 可视化
plotter = pv.Plotter()

# 绘制障碍物点云
cloud = pv.PolyData(obstacle_points)
plotter.add_points(cloud, color='b', point_size=3, render_points_as_spheres=True)

# 绘制质心和最小包围矩形
for centroid, bounding_box in zip(obstacle_centroids, bounding_boxes):
    plotter.add_mesh(pv.Sphere(radius=0.5, center=centroid), color='r')
    # 构建矩形边框
    box_edges = np.array([
        bounding_box[0], bounding_box[1],
        bounding_box[1], bounding_box[2],
        bounding_box[2], bounding_box[3],
        bounding_box[3], bounding_box[0]
    ])
    plotter.add_lines(box_edges, color='g')

plotter.show()
