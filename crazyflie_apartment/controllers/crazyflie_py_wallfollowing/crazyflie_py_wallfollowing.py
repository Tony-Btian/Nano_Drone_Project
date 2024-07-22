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

from controller import Robot
from controller import Keyboard
from controller import Supervisor
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
from ultralytics import YOLO
from image_processing import depth_estimation_and_object_recognition
from waypoint_navigation import WaypointNavigation
from astart_planner import AStarPlanner

FLYING_ATTITUDE = 1
GRID_SIZE = 10
DEPTH_THRESHOLD = 100

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

    wall_following = WallFollowing(angle_value_buffer=0.01, 
                                   reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, 
                                   init_state=WallFollowing.StateWallFollowing.FORWARD)

    autonomous_mode = False
    image_process_mode = False
    yolo_process_mode = False
    depth_process_mode = False

    # Image Processor
    Image_Processor = depth_estimation_and_object_recognition()

    # Getting the Camera's Parameters
    width = camera.getWidth()
    height = camera.getHeight()
    # fov = camera.getFov()  # Field of View
    # near = camera.getNear()

    # # Calculate the Internal Reference Matrix
    # cx = width / 2.0
    # cy = height / 2.0

    # # Assuming a Pixel Aspect Ratio of 1, the Focal Length is Converted to Pixel Units
    # fx = width / (2 * np.tan(fov / 2.0)) # Focal Length
    # fy = fx  # 对于正方形像素，fy=fx
    # K = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0,  0,  1]
    # ])
    # print("Camera intrinsic matrix:", K)
    # print("Focal length:", fx)
    # print("Width:", width)
    # print("Height:", height)
    # print("Field of View:", fov)
    # print("Near:", near)

    # Waypoint Initialize 
    # waypoints = [
    #     (0.0, 0.0, 1.0),  # Starting Point
    #     (1.0, 0.0, 1.0),  # Waypoint 1
    #     (1.0, 1.0, 1.0),  # Waypoint 2
    #     (0.0, 1.0, 1.0),  # Ending Point
    # ]

    # Waypoint Controller Instantiation
    # navigator = WaypointNavigation(waypoints)

    # AStart Planner Instantiation
    # navigator = AStarPlanner()

    #    # Define start and goal positions in grid coordinates
    start_pos = (0, 0)  # 定义起点
    goal_pos = (9, 9)   # 定义终点，假设网格大小为10x10

    print("\n")
    print("====== Controls =======\n\n")
    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n")
    print("- Use W and S to go up and down\n")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n ")


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
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

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

        # Converting images to NumPy arrays
        camera_data = camera.getImage()

        if image_process_mode:
            # Print image raw data type and size
            image_array = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))
            
            # Remove the alpha channel and make sure the image is in RGB format
            image_array = image_array[:, :, :3]
        
            # Processing images with YOLO
            # yolo_display = Image_Processor.objects_detect(image_array)

            # Image Processing with MiDas
            depth_value, depth_map = Image_Processor.estimate_depth(image_array)

            # Image pre-processing (filtering, noise reduction)
            filtered_image = Image_Processor.filter_depth_image(depth_value, method='gaussian')
            grid_map = Image_Processor.depth_to_grid(filtered_image, DEPTH_THRESHOLD, GRID_SIZE)

            # Edge Detection    
            edges_image = Image_Processor.sobel_edge_detection(filtered_image)

            # Obstacle Detection
            # point_cloud = Image_Processor.depth_to_point_cloud(filtered_image, K)

            # Convert and adjust images as needed
            images = [image_array, depth_map, filtered_image, grid_map, edges_image]
            formatted_images = Image_Processor.ensure_same_format(images)

            # Creating a three-view image
            tripple_viewer = cv2.hconcat(formatted_images)

            # Show image
            cv2.imshow('Camera Image', tripple_viewer)

            # Handling Keyboard Events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        """
        # get range in meters
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000

        # Choose a wall following direction
        # if you choose direction left, use the right range value
        # if you choose direction right, use the left range value
        direction = WallFollowing.WallFollowingDirection.LEFT
        range_side_value = range_right_value

        # Get the velocity commands from the wall following state machine
        cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = wall_following.wall_follower(
            range_front_value, range_side_value, yaw, direction, robot.getTime())

        if autonomous_mode:
            sideways_desired = cmd_vel_y
            forward_desired = cmd_vel_x
            yaw_desired = cmd_ang_w
        """

        """
        # 获取当前航点的速度命令
        if autonomous_mode:
            current_pos = (x_global, y_global, altitude)
            vx, vy, vz, all_waypoints_reached = navigator.compute_velocity_to_waypoint(current_pos)
            if all_waypoints_reached:
                # 所有航点到达后停止无人机
                autonomous_mode = False
                forward_desired = 0.0
                sideways_desired = 0.0
                yaw_desired = 0.0
            else:
                forward_desired = vx
                sideways_desired = vy
                height_diff_desired = vz - altitude
        """

        # Autonomous mode handling
        if autonomous_mode:
            pathfinder = AStarPlanner(grid_map)

            # Convert GPS coordinates to grid coordinates
            start_x = int((x_global / GRID_SIZE) % grid_map.shape[0])
            start_y = int((y_global / GRID_SIZE) % grid_map.shape[1])
            goal_x, goal_y = goal_pos

            path = pathfinder.a_star((start_x, start_y), (goal_x, goal_y))
            if path:
                next_move = path[1]
                target_x, target_y = next_move
                current_x, current_y = start_x, start_y
                forward_desired = target_x - current_x
                sideways_desired = target_y - current_y
            else:
                print("No valid path to goal.")
                autonomous_mode = False
                forward_desired = 0
                sideways_desired = 0
                yaw_desired = 0
                height_diff_desired = 0


        # PID 速度控制器（固定高度）
        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

    cv2.destroyAllWindows()
