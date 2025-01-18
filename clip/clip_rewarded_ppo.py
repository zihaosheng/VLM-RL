import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any

import numpy as np
import open_clip

from gymnasium import spaces
import torch
import torch as th
from box import Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from clip.clip_buffer import CLIPReplayBuffer, CLIPRolloutBuffer
from clip.clip_reward_model import compute_rewards, CLIPEmbed, CLIPReward

from config import CONFIG

SelfCLIPRewardedPPO = TypeVar("SelfCLIPRewardedPPO", bound="CLIPRewardedPPO")


class CLIPRewardedPPO(PPO):
    rollout_buffer: CLIPReplayBuffer

    def __init__(
            self,
            *,
            env: VecEnv,
            config: Box,
            inference_only: bool = False,
    ):
        self.config = config

        super().__init__(
            env=env,
            policy='MultiInputPolicy',
            tensorboard_log='tensorboard',
            seed=config.seed,
            **self.config.algorithm_params,
        )
        self.ep_clip_info_buffer = None  # type: Optional[deque]

        self.inference_only = inference_only
        if not self.inference_only:
            self._setup_model()
            self._load_modules()

    def _dump_logs(self) -> None:
        pass

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = CLIPRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

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

    def _compute_clip_rewards(self) -> None:
        assert self.env is not None
        assert self.ep_info_buffer is not None
        ep_info_buffer_maxlen = self.ep_info_buffer.maxlen
        assert ep_info_buffer_maxlen is not None

        frames = torch.from_numpy(np.array(self.rollout_buffer.render_arrays))  # [1024, 80, 160, 3]
        # only use the right half
        width = frames.shape[2]
        left = width // 2
        frames = frames[:, :, left:, :]
        rewards0 = compute_rewards(
            model=self.reward_model,
            frames=frames,
            batch_size=self.config.clip_reward_params.batch_size,
        )
        rewards0 = rewards0.numpy().reshape(-1, 1)
        print("Clip reward ...")
        print(list(np.round(rewards0.flatten(), 4)))

        base_reward = np.array(self.rollout_buffer.base_rewards).reshape(-1, 1)

        speeds = np.array(self.rollout_buffer.speeds)
        print("Speed ...")
        print(list(np.round(speeds.flatten(), 4)))

        thre_min, thre_max = 0.0, 0.03
        rewards0 = np.clip(rewards0, a_min=thre_min, a_max=thre_max)
        rewards0 = 1 - (rewards0 - thre_min) / (thre_max - thre_min)

        centering_factors = np.array(self.rollout_buffer.centering_factors)
        angle_factors = np.array(self.rollout_buffer.angle_factors)
        distance_std_factors = np.array(self.rollout_buffer.distance_std_factors)

        desired_speed = np.clip(rewards0.flatten(), 0.0, 1.0) * CONFIG.reward_params.target_speed
        r_speeds = 1.0 - np.abs(speeds - desired_speed) / CONFIG.reward_params.target_speed

        rewards = (r_speeds * centering_factors * angle_factors * distance_std_factors).reshape(-1, 1)
        rewards = np.where(base_reward < 0, base_reward, rewards)

        print("Final reward ...")
        print(list(np.round(rewards.flatten(), 4)))

        self.rollout_buffer.clear_render_arrays()
        self.rollout_buffer.rewards = rewards

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: CLIPRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, infos)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        if not self.inference_only:
            self._compute_clip_rewards()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

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

    def train(self) -> None:
        self._log()
        super().train()

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

    def learn(self: SelfCLIPRewardedPPO, *args, **kwargs) -> SelfCLIPRewardedPPO:
        assert not self.inference_only
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:  # type: ignore
        super().save(*args, exclude=["reward_model", "worker_frames_tensor"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfCLIPRewardedPPO],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            load_clip: bool = True,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfCLIPRewardedPPO:
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