import math
from enum import Enum


class WallFollowing():
    class StateWallFollowing(Enum):
        FORWARD = 1
        HOVER = 2
        TURN_TO_FIND_WALL = 3
        TURN_TO_ALIGN_TO_WALL = 4
        FORWARD_ALONG_WALL = 5
        ROTATE_AROUND_WALL = 6
        ROTATE_IN_CORNER = 7
        FIND_CORNER = 8

    class WallFollowingDirection(Enum):
        LEFT = 1
        RIGHT = -1

    def __init__(self, reference_distance_from_wall=0.0,
                 max_forward_speed=0.2,
                 max_turn_rate=0.5,
                 wall_following_direction=WallFollowingDirection.LEFT,
                 first_run=False,
                 prev_heading=0.0,
                 wall_angle=0.0,
                 around_corner_back_track=False,
                 state_start_time=0.0,
                 ranger_value_buffer=0.2,
                 angle_value_buffer=0.1,
                 range_lost_threshold=0.3,
                 in_corner_angle=0.8,
                 wait_for_measurement_seconds=1.0,
                 init_state=StateWallFollowing.FORWARD):
        self.reference_distance_from_wall = reference_distance_from_wall
        self.max_forward_speed = max_forward_speed
        self.max_turn_rate = max_turn_rate
        self.wall_following_direction_value = float(wall_following_direction.value)
        self.first_run = first_run
        self.prev_heading = prev_heading
        self.wall_angle = wall_angle
        self.around_corner_back_track = around_corner_back_track
        self.state_start_time = state_start_time
        self.ranger_value_buffer = ranger_value_buffer
        self.angle_value_buffer = angle_value_buffer
        self.range_threshold_lost = range_lost_threshold
        self.in_corner_angle = in_corner_angle
        self.wait_for_measurement_seconds = wait_for_measurement_seconds

        self.first_run = True
        self.state = init_state
        self.time_now = 0.0
        self.speed_redux_corner = 3.0
        self.speed_redux_straight = 2.0

    def value_is_close_to(self, real_value, checked_value, margin):
        return abs(real_value - checked_value) <= margin

    def wrap_to_pi(self, number):
        return (number + math.pi) % (2 * math.pi) - math.pi

    def command_turn(self, reference_rate):
        velocity_x = 0.0
        rate_yaw = self.wall_following_direction_value * reference_rate
        return velocity_x, rate_yaw

    def command_hover(self):
        velocity_x = 0.0
        velocity_y = 0.0
        rate_yaw = 0.0
        return velocity_x, velocity_y, rate_yaw

    def command_forward(self, forward_speed):
        velocity_x = forward_speed
        return velocity_x, 0.0

    def state_transition(self, new_state):
        self.state_start_time = self.time_now
        return new_state

    def wall_follower(self, front_range, side_range, current_heading,
                      wall_following_direction, time_outer_loop):
        self.wall_following_direction_value = float(wall_following_direction.value)
        self.time_now = time_outer_loop

        if self.first_run:
            self.prev_heading = current_heading
            self.around_corner_back_track = False
            self.first_run = False

        if self.state == self.StateWallFollowing.FORWARD:
            if front_range < self.reference_distance_from_wall + self.ranger_value_buffer:
                self.state = self.state_transition(self.StateWallFollowing.TURN_TO_FIND_WALL)
        elif self.state == self.StateWallFollowing.HOVER:
            pass
        elif self.state == self.StateWallFollowing.TURN_TO_FIND_WALL:
            side_range_check = side_range < (self.reference_distance_from_wall /
                                             math.cos(math.pi/4) + self.ranger_value_buffer)
            front_range_check = front_range < (self.reference_distance_from_wall /
                                               math.cos(math.pi/4) + self.ranger_value_buffer)
            if side_range_check and front_range_check:
                self.prev_heading = current_heading
                self.wall_angle = self.wall_following_direction_value * \
                    (math.pi/2 - math.atan(front_range / side_range) + self.angle_value_buffer)
                self.state = self.state_transition(self.StateWallFollowing.TURN_TO_ALIGN_TO_WALL)
            if side_range < self.reference_distance_from_wall + self.ranger_value_buffer and \
                    front_range > self.reference_distance_from_wall + self.range_threshold_lost:
                self.around_corner_back_track = False
                self.prev_heading = current_heading
                self.state = self.state_transition(self.StateWallFollowing.FIND_CORNER)
        elif self.state == self.StateWallFollowing.TURN_TO_ALIGN_TO_WALL:
            align_wall_check = self.value_is_close_to(
                self.wrap_to_pi(current_heading - self.prev_heading), self.wall_angle, self.angle_value_buffer)
            if align_wall_check:
                self.state = self.state_transition(self.StateWallFollowing.FORWARD_ALONG_WALL)
        elif self.state == self.StateWallFollowing.FORWARD_ALONG_WALL:
            if side_range > self.reference_distance_from_wall + self.range_threshold_lost:
                self.state = self.state_transition(self.StateWallFollowing.FIND_CORNER)
            if front_range < self.reference_distance_from_wall + self.ranger_value_buffer:
                self.prev_heading = current_heading
                self.state = self.state_transition(self.StateWallFollowing.ROTATE_IN_CORNER)
        elif self.state == self.StateWallFollowing.ROTATE_AROUND_WALL:
            if front_range < self.reference_distance_from_wall + self.ranger_value_buffer:
                self.state = self.state_transition(self.StateWallFollowing.TURN_TO_FIND_WALL)
        elif self.state == self.StateWallFollowing.ROTATE_IN_CORNER:
            check_heading_corner = self.value_is_close_to(
                math.fabs(self.wrap_to_pi(current_heading-self.prev_heading)),
                self.in_corner_angle, self.angle_value_buffer)
            if check_heading_corner:
                self.state = self.state_transition(self.StateWallFollowing.TURN_TO_FIND_WALL)
        elif self.state == self.StateWallFollowing.FIND_CORNER:
            if side_range <= self.reference_distance_from_wall:
                self.state = self.state_transition(self.StateWallFollowing.ROTATE_AROUND_WALL)
        else:
            self.state = self.state_transition(self.StateWallFollowing.HOVER)

        command_velocity_x_temp = 0.0
        command_velocity_y_temp = 0.0
        command_angle_rate_temp = 0.0

        if self.state == self.StateWallFollowing.FORWARD:
            command_velocity_x_temp = self.max_forward_speed
        elif self.state == self.StateWallFollowing.HOVER:
            command_velocity_x_temp, command_velocity_y_temp, command_angle_rate_temp = self.command_hover()
        elif self.state == self.StateWallFollowing.TURN_TO_FIND_WALL:
            command_velocity_x_temp, command_angle_rate_temp = self.command_turn(self.max_turn_rate)
        elif self.state == self.StateWallFollowing.TURN_TO_ALIGN_TO_WALL:
            command_velocity_x_temp, command_angle_rate_temp = self.command_turn(self.max_turn_rate)
        elif self.state == self.StateWallFollowing.FORWARD_ALONG_WALL:
            command_velocity_x_temp, command_velocity_y_temp = self.command_forward(self.max_forward_speed)
        elif self.state == self.StateWallFollowing.ROTATE_AROUND_WALL:
            command_velocity_x_temp, command_velocity_y_temp, command_angle_rate_temp = self.command_turn_around_corner_and_adjust(
                self.reference_distance_from_wall, side_range)
        elif self.state == self.StateWallFollowing.ROTATE_IN_CORNER:
            command_velocity_x_temp, command_angle_rate_temp = self.command_turn(self.max_turn_rate)
        elif self.state == self.StateWallFollowing.FIND_CORNER:
            command_velocity_y_temp, command_angle_rate_temp = self.command_align_corner(
                -1 * self.max_turn_rate, side_range, self.reference_distance_from_wall)
        else:
            command_velocity_x_temp, command_velocity_y_temp, command_angle_rate_temp = self.command_hover()

        return command_velocity_x_temp, command_velocity_y_temp, command_angle_rate_temp, self.state
