import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0, 10, 100)
desired_yaw = np.ones_like(time) * 30  # Desired yaw angle of 30 degrees
actual_yaw = desired_yaw * (1 - np.exp(-0.6 * time))

plt.figure(figsize=(10, 5))
plt.plot(time, desired_yaw, label='Desired Yaw')
plt.plot(time, actual_yaw, label='Actual Yaw', linestyle='--')
plt.title('Yaw Control Response')
plt.xlabel('Time (s)')
plt.ylabel('Yaw Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

time = np.linspace(0, 10, 100)
desired_rp = np.ones_like(time) * 15  # Desired roll/pitch angle of 15 degrees
actual_rp = desired_rp * (1 - np.exp(-0.8 * time))

plt.figure(figsize=(10, 5))
plt.plot(time, desired_rp, label='Desired Roll/Pitch')
plt.plot(time, actual_rp, label='Actual Roll/Pitch', linestyle='--')
plt.title('Roll and Pitch Control Response')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

time = np.linspace(0, 10, 100)
desired_altitude = np.ones_like(time) * 2  # Desired altitude of 2 meters
actual_altitude = desired_altitude * (1 - np.exp(-2.0 * time))

plt.figure(figsize=(10, 5))
plt.plot(time, desired_altitude, label='Desired Altitude')
plt.plot(time, actual_altitude, label='Actual Altitude', linestyle='--')
plt.title('Altitude Control Response')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (meters)')
plt.legend()
plt.grid(True)
plt.show()



# Example data for roll response
time = np.linspace(0, 5, 100)
roll_response = 1 - np.exp(-time) * (np.cos(2 * np.pi * time) + np.sin(2 * np.pi * time) / (2 * np.pi))

plt.figure(figsize=(10, 5))
plt.plot(time, roll_response, label='Roll Response', color='b')
plt.axhline(y=1, color='r', linestyle='--', label='Desired Setpoint')
plt.title('Roll Response to Step Input')
plt.xlabel('Time (seconds)')
plt.ylabel('Roll Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

# Example data for pitch response
pitch_response = 1 - np.exp(-time) * (np.cos(2 * np.pi * time) + np.sin(2 * np.pi * time) / (2 * np.pi))

plt.figure(figsize=(10, 5))
plt.plot(time, pitch_response, label='Pitch Response', color='g')
plt.axhline(y=1, color='r', linestyle='--', label='Desired Setpoint')
plt.title('Pitch Response to Step Input')
plt.xlabel('Time (seconds)')
plt.ylabel('Pitch Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

# Example data for yaw response
yaw_response = 1 - np.exp(-time) * (np.cos(2 * np.pi * time) + np.sin(2 * np.pi * time) / (2 * np.pi))

plt.figure(figsize=(10, 5))
plt.plot(time, yaw_response, label='Yaw Response', color='m')
plt.axhline(y=1, color='r', linestyle='--', label='Desired Setpoint')
plt.title('Yaw Response to Step Input')
plt.xlabel('Time (seconds)')
plt.ylabel('Yaw Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

# Example data for altitude response
altitude_response = 1 - np.exp(-time) * (np.cos(2 * np.pi * time) + np.sin(2 * np.pi * time) / (2 * np.pi))

plt.figure(figsize=(10, 5))
plt.plot(time, altitude_response, label='Altitude Response', color='c')
plt.axhline(y=1, color='r', linestyle='--', label='Desired Setpoint')
plt.title('Altitude Response to Step Input')
plt.xlabel('Time (seconds)')
plt.ylabel('Altitude (meters)')
plt.legend()
plt.grid(True)
plt.show()
