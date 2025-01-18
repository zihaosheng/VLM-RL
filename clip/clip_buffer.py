import torch as th
from gym import spaces
from typing import Any, Dict, List, Union
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer, DictRolloutBuffer


class CLIPReplayBuffer(DictReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.render_arrays: List[np.ndarray] = []
        self.base_rewards: List = []
        self.centering_factors: List = []
        self.angle_factors: List = []
        self.speeds: List = []
        self.distance_std_factors: List = []

    def add(
            self,
            obs: Dict[str, Any],
            next_obs: Dict[str, Any],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )
        assert len(self.render_arrays) < self.buffer_size
        self.render_arrays.append(infos[0]["render_array"])
        self.base_rewards.append(reward[0])
        self.centering_factors.append(infos[0]["centering_factor"])
        self.angle_factors.append(infos[0]["angle_factor"])
        self.speeds.append(infos[0]["speed"])
        self.distance_std_factors.append(infos[0]["distance_std_factor"])

    def clear_render_arrays(self) -> None:
        self.render_arrays = []
        self.base_rewards = []
        self.centering_factors = []
        self.angle_factors = []
        self.speeds = []
        self.distance_std_factors = []


class CLIPRolloutBuffer(DictRolloutBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )
        self.render_arrays: List[np.ndarray] = []
        self.base_rewards: List = []
        self.centering_factors: List = []
        self.angle_factors: List = []
        self.speeds: List = []
        self.distance_std_factors: List = []

    def add(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob
        )
        assert len(self.render_arrays) < self.buffer_size
        self.render_arrays.append(infos[0]["render_array"])
        self.base_rewards.append(reward[0])
        self.centering_factors.append(infos[0]["centering_factor"])
        self.angle_factors.append(infos[0]["angle_factor"])
        self.speeds.append(infos[0]["speed"])
        self.distance_std_factors.append(infos[0]["distance_std_factor"])

    def clear_render_arrays(self) -> None:
        self.render_arrays = []
        self.base_rewards = []
        self.centering_factors = []
        self.angle_factors = []
        self.speeds = []
        self.distance_std_factors = []
