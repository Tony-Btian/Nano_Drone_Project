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

# 打印部分三维坐标
print(points_3d[:10])

# 可视化或保存点云数据
# 例如，将点云保存为PLY文件
def save_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

save_ply('point_cloud.ply', points_3d)
