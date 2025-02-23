import torch as th
from box import Box
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch


class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]

        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 58, 38)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 28, 18)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 13, 8)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 6, 4)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 2)
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).view(-1).shape[0]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x


class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key == "seg_camera":
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))
        return torch.cat(encoded_tensor_list, dim=1)


algorithm_params = {
    "PPO": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])],
                           features_extractor_class=CustomMultiInputExtractor,
                           features_extractor_kwargs=dict(features_dim=256),
                           )
    ),
    "SAC": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        device="cuda:0",
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_CLIP": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(
            log_std_init=-3, net_arch=[500, 300],
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "5": ["steer", "throttle", "speed", "waypoints", "seg_camera"],
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_5_no_early_stop": dict(
        early_stop=False,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_clg": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        target_prompts=[
            "Two cars have collided with each other on the road",
            "The road is clear with no car accidents",
        ],
    ),
    "reward_lord": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        target_prompts=[
            "Two cars have collided with each other on the road",
        ],
    ),
    "reward_vlm_rm": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        alpha=0.5,
        target_prompts=[
            "A car is driving safely",
        ],
        baseline_prompts=[
            "A car",
        ],
    ),
    "reward_fn_Chen": dict(
        early_stop=True,
        min_speed=0.0,
        max_speed=28.8,
        target_speed=25.0,
        max_distance=4.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_ASAP": dict(
        early_stop=True,
        min_speed=0.0,
        max_speed=50.0,
        target_speed=30.0,
        max_distance=3.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-5,
    ),
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_vlm_rl = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_clg"],
    "vlm_reward_type": "VLM-RL",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

_CONFIG_vlm_rl_ppo = {
    "algorithm": "CLIP-PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_clg"],
    "vlm_reward_type": "VLM-RL",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "action_space_type": "discrete",
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

_CONFIG_lord = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_lord"],
    "vlm_reward_type": "LORD",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_lord_speed = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_lord"],
    "vlm_reward_type": "LORD-Speed",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_vlm_rm = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "VLM-RM",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_vlm_sr = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "VLM-SR",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_roboclip = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "RoboCLIP",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_tirl_sac = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_ppo = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_chatscene_sac = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_chatscene",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_chatscene_ppo = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_chatscene",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_revolve = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_revolve",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_revolve_auto = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_revolve_auto",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_Chen = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "vae_model": None,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_Chen",
    "reward_params": reward_params["reward_fn_Chen"],
    "obs_res": (80, 120),
    "seed": 120,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_ASAP = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "vae_model": None,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_ASAP",
    "reward_params": reward_params["reward_fn_ASAP"],
    "obs_res": (80, 120),
    "seed": 120,
    "wrappers": [],
    "use_rgb_bev": False,
}


CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "vlm_rl": _CONFIG_vlm_rl,
    "vlm_rl_ppo": _CONFIG_vlm_rl_ppo,
    "lord": _CONFIG_lord,
    "lord_speed": _CONFIG_lord_speed,
    "vlm_rm": _CONFIG_vlm_rm,
    "vlm_sr": _CONFIG_vlm_sr,
    "roboclip": _CONFIG_roboclip,
    "tirl_sac": _CONFIG_tirl_sac,
    "tirl_ppo": _CONFIG_tirl_ppo,
    "chatscene_sac": _CONFIG_chatscene_sac,
    "chatscene_ppo": _CONFIG_chatscene_ppo,
    "revolve": _CONFIG_revolve,
    "revolve_auto": _CONFIG_revolve_auto,
    "Chen": _CONFIG_Chen,
    "ASAP": _CONFIG_ASAP,
}

CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG
