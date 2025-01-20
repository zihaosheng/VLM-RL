import math

import numpy as np

from config import CONFIG


min_speed = CONFIG.reward_params.min_speed
max_speed = CONFIG.reward_params.max_speed
target_speed = CONFIG.reward_params.target_speed
max_distance = CONFIG.reward_params.max_distance
max_std_center_lane = CONFIG.reward_params.max_std_center_lane
max_angle_center_lane = CONFIG.reward_params.max_angle_center_lane
penalty_reward = CONFIG.reward_params.penalty_reward
early_stop = CONFIG.reward_params.early_stop
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            speed = env.vehicle.get_speed()
            if speed < 1.0:
                env.low_speed_timer += 1
            else:
                env.low_speed_timer = 0.0  # Reset timer if speed goes above threshold

            # Check if speed is low for 90 consecutive second
            if env.low_speed_timer >= 90 * env.fps:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance and not env.eval:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed and not env.eval:
                env.terminal_state = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            env.low_speed_timer = 0.0
            if reward_fn in {reward_fn5}:
                reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func

def reward_fn_revolve(env):
    """
    Revolve reward function
    https://arxiv.org/pdf/2406.01309
    :return: reward value and reward_components
    """
    # Parameters to tweak the importance of different reward components
    collision_penalty = -100
    inactivity_penalty = -10
    speed_reward_weight = 2.0
    position_reward_weight = 1.0
    smoothness_reward_weight = 0.5
    # Adjusted temperature parameters for score transformation
    speed_temp = 0.5  # Increased temperature for speed_reward
    position_temp = 0.1
    smoothness_temp = 0.1
    reward_components = {
        "collision_penalty": 0,
        "inactivity_penalty": 0,
        "speed_reward": 0,
        "position_reward": 0,
        "smoothness_reward": 0
    }
    collision = env.collision_state
    speed = env.vehicle.get_speed() / 3.6  # unit: m/s
    action_list = env.action_list  ### record the historical actions by setting self.steering_list = []
    min_pos = env.distance_from_center

    # Penalize for collision
    if collision:
        reward_components["collision_penalty"] = collision_penalty
    # Penalize for inactivity (speed too close to zero)
    if speed < 1.5:  # Adjusted threshold for inactivity
        reward_components["inactivity_penalty"] = inactivity_penalty
    # Reward for maintaining an optimal speed range
    if 4.0 <= speed <= 6.0:
        # Centering and normalizing around the average of optima speed range
        speed_score = 1 - np.abs(speed - 5) / 1.75
    else:
        speed_score = -1
    # Use a sigmoid function to smoothly transform the speed score and constrain it
    reward_components["speed_reward"] = speed_reward_weight * (1 / (1 + np.exp(
        -speed_score / speed_temp)))
    # Reward for being close to the center of the road
    position_score = np.exp(-min_pos / position_temp)
    reward_components["position_reward"] = position_reward_weight * position_score
    # Reward for smooth driving (small variations in consecutive steering actions)
    steering_smoothness = -np.std(action_list)
    reward_components["smoothness_reward"] = smoothness_reward_weight * np.exp(
        steering_smoothness / smoothness_temp)
    # Calculate total reward, giving precedence to penalties
    total_reward = 0
    if reward_components["collision_penalty"] < 0:
        total_reward = reward_components["collision_penalty"]
    elif reward_components["inactivity_penalty"] < 0:
        total_reward = reward_components["inactivity_penalty"]
    else:
        total_reward = sum(reward_components.values())
    # Ensure the total reward is within a reasonable range
    total_reward = np.clip(total_reward, -1, 1)
    return total_reward

reward_functions["reward_fn_revolve"] = create_reward_fn(reward_fn_revolve)

def reward_fn_revolve_auto(env):
    """
    An variant of revolve reward function, with an automatic feedback mechanism originally proposed in Eureka
    :return: reward value
    """

    def calculate_distance(point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

    def get_distance_to_nearest_obstacle(vehicle, max_distance_view=20):
        vehicle_transform = vehicle.get_transform()
        start_location = vehicle_transform.location
        forward_vector = vehicle_transform.get_forward_vector()

        end_location = start_location + forward_vector * max_distance_view

        world = vehicle.get_world()
        hit_result = world.cast_ray(start_location, end_location)

        if hit_result:
            min_distance = 9999
            for hit_res in hit_result:
                distance = calculate_distance(start_location, hit_res.location)
                if distance < min_distance:
                    min_distance = distance
            # distance = start_location.distance(hit_result.location)
            return min_distance
        else:
            return max_distance_view

    # Define temperature parameters for reward transformations
    temp_collision = 5.0
    temp_inactivity = 10.0
    temp_centering = 2.0
    temp_speed = 2.0
    temp_smoothness = 1.0
    temp_proximity = 2.0
    # Initialize the total reward and reward components dictionary
    reward_components = {
        'collision_penalty': 0,
        'inactivity_penalty': 0,
        'lane_centering_bonus': 0,
        'speed_regulation_bonus': 0,
        'smooth_driving_bonus': 0,
        'front_distance_bonus': 0
    }
    collision = env.collision_state
    speed = env.vehicle.get_speed() / 3.6  # unit: m/s
    action_list = env.action_list  ### record the historical actions by setting self.steering_list = []
    min_pos = env.distance_from_center

    # Penalty for collisions
    reward_components['collision_penalty'] = np.exp(-temp_collision) if collision else 0
    # Penalty for inactivity
    inactivity_threshold = 1.5
    if speed < inactivity_threshold and not collision:
        reward_components['inactivity_penalty'] = np.exp(-temp_inactivity * (
                inactivity_threshold - speed))
    # Lane centering bonus
    max_distance_from_center = 2.0  # adaptable based on road width
    reward_components['lane_centering_bonus'] = np.exp(-temp_centering * min_pos)
    # Speed regulation bonus
    ideal_speed = (4.0 + 6.0) / 2
    speed_range = 6.0 - 4.0
    reward_components['speed_regulation_bonus'] = np.exp(-temp_speed * (abs(speed - ideal_speed) / speed_range))
    # Smooth driving bonus
    max_steering_variance = 0.1  # configured to the acceptable steering variance for full bonus
    smoothness_factor = np.std(action_list)
    reward_components['smooth_driving_bonus'] = np.exp(-temp_smoothness * (
            smoothness_factor / max_steering_variance))
    # Front distance bonus
    max_distance_view = 20
    distance = get_distance_to_nearest_obstacle(env.vehicle, max_distance_view)
    reward_components['front_distance_bonus'] = np.clip(distance / max_distance_view, 0, 1)
    # Sum the individual rewards for the total reward
    total_reward = sum(reward_components.values())
    # Ensure penalties take precedence
    if collision or speed < inactivity_threshold:
        total_reward = -1  # Apply max negative reward for collision or inactivity
    # Normalize the total reward to lie between-1 and 1
    total_reward = np.clip(total_reward, -1, 1)
    return total_reward

reward_functions["reward_fn_revolve_auto"] = create_reward_fn(reward_fn_revolve_auto)

def reward_fn_chatscene(env):
    out_lane_thres = 4
    desired_speed = 5.5

    def get_lane_dis(waypoints, x, y):
        """
            Calculate distance from (x, y) to waypoints.
            :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
            :param x: x position of vehicle
            :param y: y position of vehicle
            :return: a tuple of the distance and the closest waypoint orientation
        """
        eps = 1e-5
        dis_min = 99999
        waypt = waypoints[0]
        for pt in waypoints:
            pt_loc = pt.transform.location
            d = np.sqrt((x - pt_loc.x) ** 2 + (y - pt_loc.y) ** 2)
            if d < dis_min:
                dis_min = d
                waypt = pt
        vec = np.array([x - waypt.transform.location.x, y - waypt.transform.location.y])
        lv = np.linalg.norm(np.array(vec)) + eps
        w = np.array([np.cos(waypt.transform.rotation.yaw / 180 * np.pi),
                      np.sin(waypt.transform.rotation.yaw / 180 * np.pi)])
        cross = np.cross(w, vec / lv)
        dis = - lv * cross
        return dis, w

    collision = env.collision_state  ### self.collision in env
    waypoints = [i[0] for i in env.route_waypoints[env.current_waypoint_index % len(env.route_waypoints): ]]
    r_collision = -1 if collision else 0

    # reward for steering:
    r_steer = -env.vehicle.get_control().steer ** 2

    # reward for out of lane
    trans = env.vehicle.get_transform()
    ego_x = trans.location.x
    ego_y = trans.location.y
    dis, w = get_lane_dis(waypoints, ego_x, ego_y)
    r_out = -1 if abs(dis) > out_lane_thres else 0

    # reward for speed tracking
    v = env.vehicle.get_velocity()

    # cost for too fast
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)
    r_fast = -1 if lspeed_lon > desired_speed else 0

    # cost for lateral acceleration
    r_lat = -abs(env.vehicle.get_control().steer) * lspeed_lon ** 2

    # combine all rewards
    total_reward = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
    return total_reward

reward_functions["reward_fn_chatscene"] = create_reward_fn(reward_fn_chatscene)

def reward_fn_simple(env):
    """
    Trustworthy safety improvement for autonomous driving using reinforcement learning, 2022
    When getting collisions, the reward is -1, else it is 0.
    """
    collision = env.collision_state  ### self.collision in env
    if collision:
        return -1
    else:
        return 0

reward_functions["reward_fn_simple"] = create_reward_fn(reward_fn_simple)

def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor

    return reward

reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_Chen(env):
    """
    A reward function that combines multiple factors:
    r = 200*r_collision + v_lon + 10*r_fast + r_out - 5*α^2 + 0.2*r_lat - 0.1

    where:
    - r_collision: -1 if collision occurs, 0 otherwise
    - v_lon: longitudinal speed of the ego vehicle
    - r_fast: -1 if speed exceeds threshold (8 m/s), 0 otherwise
    - r_out: -1 if vehicle leaves the lane, 0 otherwise
    - α: steering angle in radians, penalized quadratically
    - r_lat: lateral acceleration term, calculated as r_lat = -|α|*v_lon^2
    - -0.1: constant term to discourage the vehicle from remaining stationary
    """
    # Get vehicle state
    collision = env.collision_state
    speed = env.vehicle.get_speed() / 3.6  # Convert from km/h to m/s
    steering = env.vehicle.get_control().steer  # Steering angle in radians

    # Calculate individual reward components
    r_collision = -1 if collision else 0
    v_lon = speed  # Longitudinal speed
    r_fast = -1 if speed > 8 else 0  # Penalty for exceeding 8 m/s
    r_out = -1 if env.distance_from_center > 4 else 0  # Penalty for leaving lane (assuming 4m threshold)
    alpha_squared = -5 * (steering ** 2)  # Quadratic steering penalty
    r_lat = -abs(steering) * (speed ** 2)  # Lateral acceleration penalty
    constant_term = -0.1  # Constant penalty

    # Combine all components according to the formula
    total_reward = (200 * r_collision +
                    v_lon +
                    10 * r_fast +
                    r_out +
                    alpha_squared +
                    0.2 * r_lat +
                    constant_term)

    return total_reward


reward_functions["reward_fn_Chen"] = create_reward_fn(reward_fn_Chen)


def reward_fn_ASAP(env):
    """
    ASAP (Approximate Safe Action Policy) reward function that combines multiple factors:
    r_t = R_progress + R_destination + R_crash + R_overtaking

    where:
    - R_progress: 1 reward for every 10 meters of distance completed
    - R_destination: 1 reward if the agent successfully reaches its destination
    - R_crash: -5 penalty if the agent collides with another vehicle or road curbs
    - R_overtaking: 0.1 reward each time the agent overtakes another vehicle
    """
    # Get vehicle state
    collision = env.collision_state

    # Calculate progress reward (1 reward per 10 meters)
    distance_traveled = env.distance_traveled  # Total distance traveled in meters
    r_progress = distance_traveled / 10.0  # 1 reward per 10 meters

    # Calculate destination reward
    r_destination = 1.0 if env.success_state else 0.0

    # Calculate crash penalty
    r_crash = -5.0 if collision else 0.0

    # Calculate overtaking reward
    # Note: This requires tracking overtaken vehicles, which might need to be implemented in the environment
    # For now, we'll use a placeholder based on nearby vehicles
    r_overtaking = 0.0  # This should be implemented based on environment's vehicle tracking

    # Combine all components according to the formula
    total_reward = (r_progress +
                    r_destination +
                    r_crash +
                    r_overtaking)

    return total_reward


reward_functions["reward_fn_ASAP"] = create_reward_fn(reward_fn_ASAP)