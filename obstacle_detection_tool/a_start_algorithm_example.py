import heapq

class Node:
    def __init__(self, x, y, cost=0, heuristic=0):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.total = cost + heuristic

    def __lt__(self, other):
        return self.total < other.total

def a_star_search(grid, start, end):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(node):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x2, y2 = node.x + dx, node.y + dy
            if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]) and grid[x2][y2] == 0:
                neighbors.append((x2, y2))
        return neighbors

    start_node = Node(start[0], start[1], 0, heuristic(start, end))
    open_list = []
    heapq.heappush(open_list, start_node)
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        current = heapq.heappop(open_list)

        if (current.x, current.y) == end:
            path = []
            while (current.x, current.y) != start:
                path.append((current.x, current.y))
                current = came_from[(current.x, current.y)]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[(current.x, current.y)] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor)
                heapq.heappush(open_list, Node(neighbor[0], neighbor[1], new_cost, priority))
                came_from[neighbor] = current

    return None

# 示例网格 (0 表示通路, 1 表示障碍)
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0]
]

start = (0, 0)
end = (4, 4)

path = a_star_search(grid, start, end)
if path:
    print("Path found:", path)
else:
    print("No path found")