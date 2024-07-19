import math
import numpy as np

class WaypointNavigation:
    def __init__(self, waypoints, max_speed=0.5, threshold=0.1):
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.max_speed = max_speed
        self.threshold = threshold

    def compute_velocity_to_waypoint(self, current_pos):
        if self.current_waypoint_index >= len(self.waypoints):
            return 0.0, 0.0, 0.0, True  # All waypoints reached

        target_pos = self.waypoints[self.current_waypoint_index]
        direction_vector = np.array(target_pos) - np.array(current_pos)
        distance = np.linalg.norm(direction_vector)
        if distance < self.threshold:  # Check if the waypoint is reached
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                return 0.0, 0.0, 0.0, True  # All waypoints reached
            target_pos = self.waypoints[self.current_waypoint_index]
            direction_vector = np.array(target_pos) - np.array(current_pos)
            distance = np.linalg.norm(direction_vector)

        velocity = (direction_vector / distance) * self.max_speed
        return velocity[0], velocity[1], velocity[2], False