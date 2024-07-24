import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取深度图
depth_map_path = 'depth_image.png'
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

# 检查深度图是否成功读取
if depth_map is None:
    raise ValueError(f"无法读取深度图文件 '{depth_map_path}'，请检查文件格式和内容。")

# 检查图像通道数
if len(depth_map.shape) == 3:
    # 如果是多通道图像，提取第一个通道作为深度图
    depth_map = depth_map[:, :, 0]

# 将深度图归一化到0-255范围
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 使用Canny边缘检测
edges = cv2.Canny(depth_map_normalized, 50, 150)

# 显示原始深度图和边缘图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Depth Map')
plt.imshow(depth_map, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Edge Map')
plt.imshow(edges, cmap='gray')
plt.show()
