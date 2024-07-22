import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]


    def heuristic(self, a, b):
        # 使用曼哈顿距离作为启发式函数
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    

    def get_neighbors(self, node):
        neighbors = [(node[0]+1, node[1]), (node[0]-1, node[1]),
                     (node[0], node[1]+1), (node[0], node[1]-1)]
        valid_neighbors = []
        for neighbor in neighbors:
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                if self.grid[neighbor[0], neighbor[1]] == 0:
                    valid_neighbors.append(neighbor)
        return valid_neighbors
    

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 如果没有路径