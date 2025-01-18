import os
import argparse
import pandas as pd
import numpy as np
import config
from clip.clip_rewarded_ppo import CLIPRewardedPPO

parser = argparse.ArgumentParser(description="Eval a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2020, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--model", type=str, default="./model_400000_steps.zip", help="Path to a model evaluate")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--no_record_video", action="store_false", help="If True, record video of the evaluation")
parser.add_argument("--config", type=str, default="vlm_rl", help="Config to use (default: vlm_rl)")
parser.add_argument("--seed", type=int, default=101, help="random seed")
parser.add_argument("--device", type=str, default="cuda:0", help="cpu, cuda:0, cuda:1, cuda:2")
parser.add_argument("--density", choices=['empty', 'regular', 'dense'], default="regular",
                    help="different traffic densities")
parser.add_argument("--town", choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05'], default="Town02",
                    help="different traffic densities")

args = vars(parser.parse_args())
CONFIG = config.set_config(args["config"])
CONFIG.seed = args["seed"]
CONFIG.algorithm_params.device = args["device"]

from stable_baselines3 import PPO, DDPG, SAC
from clip.clip_rewarded_sac import CLIPRewardedSAC

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions

from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval


def convert_state(state):
    c_state = dict()
    c_state['seg_camera'] = np.transpose(state['seg_camera'], (2, 0, 1))
    c_state['seg_camera'] = np.array([c_state['seg_camera']])
    c_state['waypoints'] = np.array([state['waypoints']])
    c_state['vehicle_measures'] = np.array([state['vehicle_measures']])
    return c_state


def run_eval(env, model, model_path=None, record_video=False, eval_suffix=''):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval{}'.format(eval_suffix))
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    state = env.reset()

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y",
               "route_x", "route_y", "routes_completed", "collision_speed", "collision_interval", "CPS", "CPM"
               ]
    df = pd.DataFrame(columns=columns)

    # Init video recording
    if record_video:
        rendered_frame = env.render(mode="rgb_array")
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    # While non-terminal state
    print("Episode ", episode_idx)
    saved_route = False
    while episode_idx < 10:
        env.extra_info.append("Evaluation")
        action, _states = model.predict(state, deterministic=True)
        next_state, reward, dones, info = env.step(action)

        state = next_state
        if env.step_count >= 150 and env.current_waypoint_index == 0:
            dones = True

        # Save route at the beginning of the episode
        if not saved_route:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())
            # Save the route to plot them later
            for way in env.route_waypoints:
                route_relative = get_displacement_vector(initial_vehicle_location,
                                                         vector(way[0].transform.location),
                                                         initial_heading)
                new_row = pd.DataFrame([['route', env.episode_idx, route_relative[0], route_relative[1]]],
                                       columns=["model_id", "episode", "route_x", "route_y"])
                df = pd.concat([df, new_row], ignore_index=True)
            saved_route = True

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)

        if env.collision_state:
            collision_speed, collision_interval, cps, cpm = env.collision_speed, env.collision_interval, env.cps, env.cpm
        else:
            collision_speed, collision_interval, cps, cpm = 0, None, 0, 0
        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_relative[0], waypoint_relative[1], None, None,
              env.routes_completed, collision_speed, collision_interval, cps, cpm
              ]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        if record_video:
            # Add frame
            rendered_frame = env.render(mode="rgb_array")
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()
            episode_idx += 1
            saved_route = False
            print("Episode ", episode_idx)

    # Release video
    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    plot_eval([csv_path])
    summary_eval(csv_path)


if __name__ == "__main__":

    model_ckpt = args["model"]
    algorithm_dict = {
        "PPO": PPO,
        "DDPG": DDPG,
        "SAC": SAC,
        "CLIP-SAC": CLIPRewardedSAC,
        "CLIP-PPO": CLIPRewardedPPO,
    }
    if CONFIG.algorithm not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    AlgorithmRL = algorithm_dict[CONFIG.algorithm]

    observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
    action_space_type = 'continuous' if CONFIG.action_space_type != 'discrete' else 'discrete'

    eval_suffix = ''
    if args['density'] == 'empty':
        activate_traffic_flow = False
        tf_num = 0
        eval_suffix += 'empty'
    else:
        activate_traffic_flow = True
        if args['density'] == 'regular':
            tf_num = 20
        else:
            tf_num = 40
            eval_suffix += 'dense'
    if args['town'] != 'Town02':
        eval_suffix += args['town']

    env = CarlaRouteEnv(obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
                        reward_fn=reward_functions[CONFIG.reward_fn], observation_space=observation_space,
                        encode_state_fn=encode_state_fn, fps=args["fps"], action_smoothing=CONFIG.action_smoothing,
                        eval=True, action_space_type=action_space_type, activate_spectator=True, activate_render=True,
                        activate_bev=True, activate_seg_bev=CONFIG.use_seg_bev, start_carla=True,
                        activate_traffic_flow=activate_traffic_flow, tf_num=tf_num, town=args["town"])

    for wrapper_class_str in CONFIG.wrappers:
        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)

    # Load the model based on the algorithm type
    if CONFIG.algorithm == "CLIP-SAC":
        model = CLIPRewardedSAC.load(model_ckpt, env=env, config=CONFIG, device=args["device"], load_clip=False)
        model.inference_only = True
    elif CONFIG.algorithm == "CLIP-PPO":
        model = CLIPRewardedPPO.load(model_ckpt, env=env, config=CONFIG, device=args["device"], load_clip=False)
        model.inference_only = True
    else:
        model = AlgorithmRL.load(model_ckpt, env=env, device=args["device"])

    print("Model loaded successfully...")

    run_eval(env, model, model_ckpt, record_video=args["no_record_video"], eval_suffix=eval_suffix)

    env.close()
