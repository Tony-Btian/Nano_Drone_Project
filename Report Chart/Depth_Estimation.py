import matplotlib.pyplot as plt
import numpy as np

# Example data: simulated depth values for visualization
depth_values_indoor = np.random.uniform(low=0.5, high=5.0, size=(480, 480))

plt.figure(figsize=(10, 5))
plt.imshow(depth_values_indoor, cmap='plasma_r')
plt.title('Indoor Environment Depth Map')
plt.colorbar(label='Depth (meters)')
plt.show()

# Example data: simulated depth values for visualization
depth_values_outdoor = np.random.uniform(low=1.0, high=20.0, size=(480, 480))

plt.figure(figsize=(10, 5))
plt.imshow(depth_values_outdoor, cmap='plasma_r')
plt.title('Outdoor Environment Depth Map')
plt.colorbar(label='Depth (meters)')
plt.show()
