import os
import time
import argparse
from typing import Any, Dict

import torch
import numpy as np
import gymnasium as gym
import retro

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integrations")
)


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, save_path: str, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param save_path: Path to save resulting videos
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.save_path = save_path

    def _init_callback(self) -> None:
        # create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            video_path = os.path.join(
                self.save_path,
                f"video_{self.num_timesteps}_steps.mp4"
            )

            self.logger.record(
                video_path,
                Video(torch.from_numpy(np.asarray([screens])), fps=50),
                exclude=("stdout", "log", "json", "csv"),
            )

        return True


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class AsterixDiscretizer(Discretizer):
    """
    Use Asterix-specific discrete actions
    based on https://github.com/farama-foundation/stable-retro/blob/master/retro/examples/discretizer.py
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [None], # no action, wait
                ["A"], # jump
                ["B"], # hit
                # ["UP"], # useless by itself
                ["DOWN"], # duck or enter an underground
                ["LEFT"],
                ["RIGHT"],
                ["UP", "B"], # throw item far away
                ["DOWN", "B"], # throw item close
                ["LEFT", "A"], # jump left
                ["RIGHT", "A"], # jump right
                ["LEFT", "B"], # attack left (useful after jump)
                ["RIGHT", "B"], # attack right (useful after jump)
            ],
        )


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n=4):
        super().__init__(env)
        self._n = n 

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._n):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


def make_retro(game, state=None, **kwargs):  
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = Monitor(env)
    env = AsterixDiscretizer(env)
    env = FrameSkip(env, n=4)    
    env = WarpFrame(env)
    # env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="CustomAsterix-Sms")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--render_mode", default="rgb_array")
    parser.add_argument("--n_envs", default=8, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log_dir", default='./logs/')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    def make_env():
        env = make_retro(
            game=args.game,
            state=args.state,
            scenario=args.scenario,
            inttype=retro.data.Integrations.ALL,
            render_mode=args.render_mode
        )
        return env

    venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv([make_env] * args.n_envs),
            n_stack=4
        )
    )

    model = DQN(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,        
        batch_size=32,        
        gamma=0.99,        
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    if args.checkpoint:
        model.set_parameters(args.checkpoint)

    callback_frequency = 10000
    checkpoint_callback = CheckpointCallback(
        save_path=args.log_dir,
        save_freq=callback_frequency
    )

    # args.render_mode = 'human'
    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        n_eval_episodes=1,
        eval_freq=callback_frequency,
        deterministic=True,
        render=True,
        verbose=1
    )

    # record_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))
    video_callback = VideoRecorderCallback(
        eval_env,
        save_path=args.log_dir,
        render_freq=callback_frequency,
        n_eval_episodes=1,
        deterministic=True
    )

    callback = CallbackList([
        eval_callback,
        checkpoint_callback,
        video_callback
    ])

    model.learn(
        total_timesteps=200_000,
        log_interval=1,
        callback=callback
    )

    model.save(args.log_dir + "/DQN_last_" + time.strftime("%Y%m%d-%H%M%S"))



if __name__ == "__main__":
    main()
