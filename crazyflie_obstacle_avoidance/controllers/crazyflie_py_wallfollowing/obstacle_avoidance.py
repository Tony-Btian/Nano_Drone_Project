"""
File: obstacle_avoidance.py

The graphics captured by UAV were converted into visual processing

Author: Binhan Tian (University of Glasgow)
Date: 25 July, 2024
"""
import numpy as np
import pyvista as pv
import time

class PointCloudVisualizer:
    def __init__(self, K):
        self.K = K
        self.plotter = pv.Plotter()
        self.point_cloud_actor = None


    def depth_to_point_cloud(self, depth_map):
        # 获取内参矩阵的参数
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        height, width = depth_map.shape

        # 创建 (i, j) 坐标网格
        j, i = np.meshgrid(np.arange(width), np.arange(height))

        z = depth_map
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy

        # 堆叠并重塑以获取点云
        point_cloud = np.dstack((x, y, z)).reshape(-1, 3)

        return point_cloud


    def visualize_point_cloud(self, depth_map):
        # 转换深度图为点云
        point_cloud = self.depth_to_point_cloud(depth_map)
        
        # 创建 PyVista 的点云对象
        point_cloud_pv = pv.PolyData(point_cloud)

        if self.point_cloud_actor is None:
            # 初次可视化
            self.point_cloud_actor = self.plotter.add_points(point_cloud_pv, render_points_as_spheres=True, point_size=2)
            self.plotter.show(auto_close=False)
        else:
            # 更新点云数据
            self.point_cloud_actor.points = point_cloud
            self.plotter.update()
            self.plotter.render()


    def update_point_cloud(self, depth_map):
        if self.point_cloud_actor is None:
            # 如果点云还没有初始化，初始化它
            self.visualize_point_cloud(depth_map)
        else:
            # 更新点云
            point_cloud = self.depth_to_point_cloud(depth_map)
            self.point_cloud_actor.points = point_cloud
            self.plotter.update()
            self.plotter.render()