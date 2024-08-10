import matplotlib.pyplot as plt
import numpy as np

# Simulated environment grid (1: obstacle, 0: free space)
environment = np.zeros((20, 20))
environment[5:15, 10] = 1
environment[10, 5:15] = 1

# Simulated path coordinates
path_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
path_y = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure(figsize=(10, 10))
plt.imshow(environment, cmap='gray')
plt.plot(path_y, path_x, marker='o', color='r', label='Planned Path')
plt.title('Path Planning in Indoor Environment')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

# Simulated environment grid (1: obstacle, 0: free space)
environment = np.zeros((20, 20))
environment[5:15, 8] = 1
environment[5:15, 12] = 1
environment[8, 5:15] = 1
environment[12, 5:15] = 1

# Simulated path coordinates
path_x = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
path_y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

plt.figure(figsize=(10, 10))
plt.imshow(environment, cmap='gray')
plt.plot(path_y, path_x, marker='o', color='r', label='Planned Path')
plt.title('Path Planning in Outdoor Environment')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

# Example data for path length and computation time
scenarios = ['Indoor Sparse', 'Indoor Cluttered', 'Outdoor Sparse', 'Outdoor Cluttered']
path_lengths = [10.2, 13.8, 11.5, 14.2]
computation_times = [40, 60, 45, 55]

plt.figure(figsize=(10, 5))
plt.bar(scenarios, path_lengths, color='b', alpha=0.7, label='Path Length (meters)')
plt.xlabel('Scenarios')
plt.ylabel('Path Length (meters)')
plt.title('Average Path Length in Different Scenarios')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(scenarios, computation_times, color='g', alpha=0.7, label='Computation Time (ms)')
plt.xlabel('Scenarios')
plt.ylabel('Computation Time (ms)')
plt.title('Average Computation Time in Different Scenarios')
plt.legend()
plt.show()
