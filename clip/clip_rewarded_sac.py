import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any

import numpy as np
import open_clip
import stable_baselines3.common.noise as sb3_noise
import torch
from box import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from clip.clip_buffer import CLIPReplayBuffer
from clip.clip_reward_model import compute_rewards, CLIPEmbed, CLIPReward

from config import CONFIG

SelfCLIPRewardedSAC = TypeVar("SelfCLIPRewardedSAC", bound="CLIPRewardedSAC")


class CLIPRewardedSAC(SAC):
    replay_buffer: CLIPReplayBuffer

    def __init__(
        self,
        *,
        env: VecEnv,
        config: Box,
        inference_only: bool = False,
    ):
        self.config = config

        if config.action_noise:
            mean = config.action_noise.mean * np.ones(env.action_space.shape)
            sigma = config.action_noise.sigma * np.ones(env.action_space.shape)
            if config.action_noise.name == "NormalActionNoise":
                action_noise = sb3_noise.NormalActionNoise(mean=mean, sigma=sigma)
            elif config.action_noise.name == "OrnsteinUhlenbeckActionNoise":
                action_noise = sb3_noise.OrnsteinUhlenbeckActionNoise(
                    mean=mean,
                    sigma=sigma,
                    theta=config.action_noise.theta,
                    dt=config.action_noise.dt,
                )
            else:
                raise ValueError(
                    f"Unknown action noise name: {config.action_noise.name}"
                )
        else:
            action_noise = None

        super().__init__(
            env=env,
            policy='MultiInputPolicy',
            replay_buffer_class=CLIPReplayBuffer,
            tensorboard_log='tensorboard',
            seed=config.seed,
            action_noise=action_noise,
            **self.config.algorithm_params,
        )
        self.ep_clip_info_buffer = None  # type: Optional[deque]

        self.inference_only = inference_only
        if not self.inference_only:
            self._load_modules()
            self.previous_num_timesteps = 0
            self.previous_num_episodes = 0

    def _dump_logs(self) -> None:
        pass

    def _load_modules(self):
        model_name_prefix, pretrained = self.config.clip_reward_params.pretrained_model.split("/")
        clip_model = open_clip.create_model(
            model_name=model_name_prefix, pretrained=pretrained  # , cache_dir=cache_dir
        )
        clip_model = CLIPEmbed(clip_model)
        target_prompts = CLIPReward.tokenize_prompts(self.config.clip_reward_params.target_prompts)
        baseline_prompts = CLIPReward.tokenize_prompts(self.config.clip_reward_params.baseline_prompts)

        clip_model = CLIPReward(
            model=clip_model,
            alpha=self.config.clip_reward_params.alpha,
            target_prompts=target_prompts,
            baseline_prompts=baseline_prompts,
        )
        self.reward_model = clip_model.eval().to(self.device)

    def _compute_clip_rewards(self):
        assert self.env is not None

        replay_buffer_pos = self.replay_buffer.pos
        total_timesteps = self.num_timesteps - self.previous_num_timesteps
        env_episode_timesteps = total_timesteps // self.env.num_envs

        frames = torch.from_numpy(np.array(self.replay_buffer.render_arrays))  # [64, height, width, 3]
        # only use the right half
        width = frames.shape[2]
        left = width // 2
        frames = frames[:, :, left:, :]
        rewards0 = compute_rewards(
            model=self.reward_model,
            frames=frames,
            batch_size=self.config.clip_reward_params.batch_size,
            vlm_reward_type=self.config.vlm_reward_type,
        )
        rewards0 = rewards0.numpy().reshape(-1, 1)
        print("Clip reward ...")
        print(list(np.round(rewards0.flatten(), 4)))
        base_reward = np.array(self.replay_buffer.base_rewards).reshape(-1, 1)
        speeds = np.array(self.replay_buffer.speeds)
        print("Speed ...")
        print(list(np.round(speeds.flatten(), 4)))

        if self.config.vlm_reward_type == "VLM-RL":
            thre_min, thre_max = 0.0, 0.03
            rewards0 = np.clip(rewards0, a_min=thre_min, a_max=thre_max)
            rewards0 = 1 - (rewards0 - thre_min) / (thre_max - thre_min)
            centering_factors = np.array(self.replay_buffer.centering_factors)
            angle_factors = np.array(self.replay_buffer.angle_factors)
            distance_std_factors = np.array(self.replay_buffer.distance_std_factors)
            desired_speed = np.clip(rewards0.flatten(), 0.0, 1.0) * CONFIG.reward_params.target_speed
            r_speeds = 1.0 - np.abs(speeds - desired_speed) / CONFIG.reward_params.target_speed
            rewards = (r_speeds * centering_factors * angle_factors * distance_std_factors).reshape(-1, 1)
            rewards = np.where(base_reward < 0, base_reward, rewards)
        elif self.config.vlm_reward_type == "LORD-Speed":
            lord_speed_r = 1.0 - np.abs(speeds - 20.0) / 20.0
            rewards = rewards0 + lord_speed_r.reshape(-1, 1)
        elif self.config.vlm_reward_type == "VLM-SR":
            rewards = np.where(rewards0 > 0.32, 1, -1)
        else:
            rewards = rewards0

        print("Final reward ...")
        print(list(np.round(rewards.flatten(), 4)))
        self.replay_buffer.clear_render_arrays()

        if not self.inference_only:
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.replay_buffer.rewards[
                replay_buffer_pos - env_episode_timesteps: replay_buffer_pos
                ] = rewards
            else:
                self.replay_buffer.rewards[
                -(env_episode_timesteps - replay_buffer_pos):
                ] = rewards[: env_episode_timesteps - replay_buffer_pos]
                self.replay_buffer.rewards[:replay_buffer_pos] = rewards[
                                                                 env_episode_timesteps - replay_buffer_pos:
                                                                 ]

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        rollout = super().collect_rollouts(*args, **kwargs)
        if not self.inference_only:
            self._compute_clip_rewards()
            self.previous_num_timesteps = self.num_timesteps
            self.previous_num_episodes = self._episode_num
        return rollout

    def _log(self) -> None:
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_gt_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self._log()
        super().train(gradient_steps, batch_size)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        *args,
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            *args,
        )
        if self.ep_clip_info_buffer is None or reset_num_timesteps:
            self.ep_clip_info_buffer = deque(maxlen=100)
        return total_timesteps, callback

    def learn(self: SelfCLIPRewardedSAC, *args, **kwargs) -> SelfCLIPRewardedSAC:
        assert not self.inference_only
        self.previous_num_timesteps = 0
        self.previous_num_episodes = 0
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:  # type: ignore
        super().save(*args, exclude=["reward_model", "worker_frames_tensor"], **kwargs)


    @classmethod
    def load(
            cls: Type[SelfCLIPRewardedSAC],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            load_clip: bool = True,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfCLIPRewardedSAC:
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                    "net_arch" in data["policy_kwargs"]
                    and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                        saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
                "policy_kwargs" in kwargs
                and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, "
                f"specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify "
                "new environments."
            )

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(
                data[key]
            )  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated.
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for
            # predict)
            if "env" in data:
                env = data["env"]

        # Ensure the config has the necessary structure
        if "config" not in data:
            data["config"] = Box(default_box=True)
        if not hasattr(data["config"], "action_noise"):
            data["config"].action_noise = None

        # pytype: disable=not-instantiable,wrong-keyword-args
        # use current device
        data["config"].algorithm_params.device = device
        model = cls(
            env=env,
            config=data["config"],
            inference_only=not load_clip,
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                    e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for "
                    f"more info). Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward
                # compatibility). This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is
                # defined, otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()

        if load_clip:
            model._load_modules()
        return model