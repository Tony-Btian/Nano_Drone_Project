import numpy as np
import threading
from math import atan2, cos, sin, pi

class SLAM:
    def __init__(self, height, width, resolution, start_pos, map_size):
        self.height = height
        self.width = width
        self.resolution = resolution
        self.start_pos = start_pos
        self.map_size = map_size
        self.slam_map = np.zeros(map_size)
        self.depth_value = np.zeros((height, width))
        self.yaw = 0.0
        self.thread = threading.Thread(target=self.update_slam_map)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def update_slam_map(self):
        while True:
            for y in range(self.height):
                for x in range(self.width):
                    distance = self.depth_value[y, x]
                    if distance < 2.0:  # 只处理一定距离内的点
                        angle = atan2(y - self.height // 2, x - self.width // 2) + self.yaw
                        map_x = int(self.start_pos[0] + (distance * cos(angle)) / self.resolution)
                        map_y = int(self.start_pos[1] + (distance * sin(angle)) / self.resolution)
                        if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                            self.slam_map[map_x, map_y] = 1
            print("SLAM Map Updated")

    def update_depth_data(self, depth_value, yaw):
        self.depth_value = depth_value
        self.yaw = yaw

    def get_slam_map(self):
        return self.slam_map