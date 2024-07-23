import threading
import cv2
import numpy as np
from astart_planner import AStarPlanner
from image_processing import depth_estimation_and_object_recognition


class ImageProcessingThread(threading.Thread):
    def __init__(self, camera, image_processor, depth_threshold, grid_size, goal_pos):
        threading.Thread.__init__(self)
        self.camera = camera
        self.image_processor = image_processor
        self.depth_threshold = depth_threshold
        self.grid_size = grid_size
        self.goal_pos = goal_pos
        self.start_x = 0
        self.start_y = 0
        self.path = []
        self.running = True

    def run(self):
        while self.running:
            if self.camera:
                # Converting images to NumPy arrays
                image_array = np.frombuffer(self.camera, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
                image_array = image_array[:, :, :3]  # Remove the alpha channel
                
                # Image Processing with MiDas
                depth_value, depth_map = self.image_processor.estimate_depth(image_array)
                filtered_image = self.image_processor.filter_depth_image(depth_value, method='gaussian')
                grid_map = self.image_processor.depth_to_grid(filtered_image, self.depth_threshold, self.grid_size)
                
                # Pathfinding
                pathfinder = AStarPlanner(grid_map)
                goal_x = int(self.goal_pos[0] / self.grid_size)
                goal_y = int(self.goal_pos[1] / self.grid_size)
                self.path = pathfinder.a_star((self.start_x, self.start_y), (goal_x, goal_y))

                # Display images for debugging
                edges_image = self.image_processor.sobel_edge_detection(filtered_image)
                images = [image_array, depth_map, filtered_image, edges_image]
                formatted_images = self.image_processor.ensure_same_format(images)
                tripple_viewer = cv2.hconcat(formatted_images)
                cv2.imshow('Camera Image', tripple_viewer)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()