# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
# MIT License
# Copyright (c) 2023 Bitcraze

"""
file: crazyflie_py_wallfollowing.py

Controls the crazyflie and implements a wall following method in webots in Python

Author:   Kimberly McGuire (Bitcraze AB)
"""

import cv2
import numpy as np
import threading
import queue

from controller import Robot
from controller import Keyboard
from math import atan2, cos, pi, sin, sqrt
from pid_controller import pid_velocity_fixed_height_controller
from ultralytics import YOLO
from image_processing import depth_estimation_and_object_recognition
from obstacle_avoidance import PointCloudVisualizer

FLYING_ATTITUDE = 1
GRID_SIZE = 10
DEPTH_THRESHOLD = 200

image_process_mode = False
image_queue = queue.Queue()

# --------------------- Image Processing ----------------------- #

def image_processing_thread_func(camera_data, height, width, K, image_processor):
    """
    """
    global image_process_mode
    while True:
        if image_process_mode:
            camera_data = camera.getImage()
            image_array = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))
            image_array = image_array[:, :, :3]

            # Image Processing with MiDas and YOLO
            depth_value, depth_map = image_processor.estimate_depth(image_array)

            # Image pre-processing (filtering, noise reduction)
            filtered_image = image_processor.filter_depth_image(depth_value, method='gaussian')
            normalized = image_processor.normalize_depth_image(filtered_image)
            # point_cloud = image_processor.depth_to_point_cloud(normalized, K)
            edges_map = image_processor.sobel_edge_detection(normalized)  # Edge Detection

            # Creating a three-view image
            images = [depth_map, normalized, edges_map]
            formatted_images = image_processor.ensure_same_format(images)  # Convert and adjust images as needed
            tripple_viewer = cv2.hconcat(formatted_images)
            cv2.imshow('Camera Image', tripple_viewer)  # Show image

            # Handling Keyboard Events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    autonomous_mode = False
    image_process_mode = False
    yolo_process_mode = False
    depth_process_mode = False

    # Image Processor
    image_processor = depth_estimation_and_object_recognition()

    # Getting the Camera's Parameters
    width = camera.getWidth()
    height = camera.getHeight()
    fov = camera.getFov()  # Field of View
    near = camera.getNear()

    # Calculate the Internal Reference Matrix
    cx = width / 2.0
    cy = height / 2.0

    # Assuming a Pixel Aspect Ratio of 1, the Focal Length is Converted to Pixel Units
    fx = width / (2 * np.tan(fov / 2.0)) # Focal Length
    fy = fx  # For square pixels, fy = fx
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    visualizer = PointCloudVisualizer(K)

    # Define start and goal positions in grid coordinates
    start_pos = [-0.9009894214148105, -5.592220508161193, 1.0013359608992145]
    goal_pos = [-8.802181198436783, -0.9554684815717253, 1.0000519299181825]

    # Define Environment Parameters
    plane_size = (9.9, 6.6) # Flat dimensions (meters)
    resolution = 0.1  # Size of each raster cell (meters)
    map_size = (int(plane_size[0] / resolution), int(plane_size[1] / resolution))
    slam_map = np.zeros(map_size)
    
    
    print("\n")
    print("====== Controls =======\n\n")
    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n")
    print("- Use W and S to go up and down\n")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n ")

    # Start image processing thread
    image_processing_thread = threading.Thread(target=image_processing_thread_func, args=(camera, height, width, K, image_processor))
    image_processing_thread.daemon = True  # Allow thread to exit when the main program exits
    image_processing_thread.start()

    # Main loop:
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time
        actual_state = {}

        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # Get sensor data
        roll  = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw   = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        y_global = gps.getValues()[1]
        altitude = gps.getValues()[2]
        current_position = gps.getValues()
        # print("GPS Value", current_position)

        v_x_global = (x_global - past_x_global)/dt
        v_y_global = (y_global - past_y_global)/dt

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
            elif key == ord('A'):
                if autonomous_mode is False:
                    autonomous_mode = True
                    print("Autonomous mode: ON")
            elif key == ord('D'):
                if autonomous_mode is True:
                    autonomous_mode = False
                    print("Autonomous mode: OFF")
            elif key == ord('N'):
                if image_process_mode is False:
                    image_process_mode = True
                    print("Image process mode: ON")
            elif key == ord('M'):
                if image_process_mode is True:
                    image_process_mode = False
                    print("Image process mode: OFF")
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt


        # ----------------- Autonomous mode handling ------------------- #

        if autonomous_mode:
            # Calculate the distance and direction to the goal
            dx = goal_pos[0] - x_global
            dy = goal_pos[1] - y_global
            distance = sqrt(dx**2 + dy**2)

            if distance > 0.5:
                angle_to_goal = atan2(dy, dx)
                desired_yaw = angle_to_goal
                yaw_error = desired_yaw - yaw

                if yaw_error > pi:
                    yaw_error -= 2 * pi
                elif yaw_error < -pi:
                    yaw_error += 2 * pi

                yaw_desired = yaw_error

                # 使用 SLAM 地图更新避障信息
                left_sum = np.sum(slam_map[:map_size[0]//2, :])
                right_sum = np.sum(slam_map[map_size[0]//2:, :])

                # 检查是否存在障碍物
                obstacle_detected = left_sum > 0 or right_sum > 0

                if obstacle_detected:
                    # 根据 SLAM 地图信息调整侧向运动
                    if left_sum > right_sum:
                        sideways_desired = -1  # 向右移动
                    else:
                        sideways_desired = 1  # 向左移动

                    forward_desired = 0  # 停止前进进行避障
                    cumulative_sideways += sideways_desired  # 记录侧向移动量
                else:
                    # 没有检测到障碍物，检查是否需要回到原轨道
                    if abs(cumulative_sideways) > 0.2:  # 阈值可以根据需要调整
                        sideways_desired = -cumulative_sideways  # 反向移动
                        forward_desired = 0  # 停止前进以调整位置
                        cumulative_sideways += sideways_desired  # 更新累计侧向移动量
                    else:
                        sideways_desired = 0
                        forward_desired = 0.5  # 沿着直线行进到终点
                                
            else:
                forward_desired = 0
                sideways_desired = 0
                yaw_desired = 0
                cumulative_sideways = 0  # 重置累计侧向移动量
                print("Reached the goal.")

            # for y in range(height):
            #         for x in range(width):
            #             distance = depth_value[y, x]
            #             if distance < 2.0:  # 只处理一定距离内的点
            #                 angle = atan2(y - height // 2, x - width // 2) + yaw
            #                 map_x = int(start_pos[0] + (distance * cos(angle)) / resolution)
            #                 map_y = int(start_pos[1] + (distance * sin(angle)) / resolution)
            #                 if 0 <= map_x < map_size[0] and 0 <= map_y < map_size[1]:
            #                     slam_map[map_x, map_y] = 1
            # print("SLAM", slam_map)


        # ------------------- PID 速度控制器（固定高度） ------------------- #

        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity( motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity( motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

    cv2.destroyAllWindows()
