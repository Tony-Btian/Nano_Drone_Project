import matplotlib.pyplot as plt
import numpy as np

# 示例数据：特征点匹配精度（匹配误差）
without_depth = np.random.normal(1.0, 0.2, 100)
with_depth = np.random.normal(0.5, 0.1, 100)

plt.figure(figsize=(10, 5))
plt.plot(without_depth, label='Without Depth Map')
plt.plot(with_depth, label='With Depth Map', linestyle='--')
plt.title('Feature Point Matching Accuracy')
plt.xlabel('Match Index')
plt.ylabel('Matching Error')
plt.legend()
plt.grid(True)
plt.show()

# 示例数据：累积漂移误差
time = np.linspace(0, 100, 100)
drift_without_depth = np.cumsum(np.random.normal(0.02, 0.01, 100))
drift_with_depth = np.cumsum(np.random.normal(0.01, 0.005, 100))

plt.figure(figsize=(10, 5))
plt.plot(time, drift_without_depth, label='Without Depth Map')
plt.plot(time, drift_with_depth, label='With Depth Map', linestyle='--')
plt.title('Cumulative Drift Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Drift (meters)')
plt.legend()
plt.grid(True)
plt.show()


# 示例数据：定位误差
localization_error_without_depth = np.random.normal(0.05, 0.02, 100)
localization_error_with_depth = np.random.normal(0.03, 0.01, 100)

plt.figure(figsize=(10, 5))
plt.plot(localization_error_without_depth, label='Without Depth Map')
plt.plot(localization_error_with_depth, label='With Depth Map', linestyle='--')
plt.title('Localization Accuracy')
plt.xlabel('Time (seconds)')
plt.ylabel('Localization Error (meters)')
plt.legend()
plt.grid(True)
plt.show()
