import os
import time
import argparse
from typing import Any, Dict

import cv2
import numpy as np
import gymnasium as gym
import retro

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
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
            self._eval_env.reset()
            screen = self._eval_env.render(mode="rgb_array")
            height = screen.shape[0]
            width = screen.shape[1]
            frameSize = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            video_path = os.path.join(
                self.save_path,
                f"video_{self.num_timesteps}_steps.mp4"
            )

            out = cv2.VideoWriter(video_path, fourcc, 50 / 4, frameSize)

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                out.write(screen)

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            out.release()

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
                # [None], # no action, wait
                ["A"], # jump
                ["B"], # hit
                ["UP"], # enter a door
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


def make_retro(game, state=None, monitor_filename=None, **kwargs):  
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = Monitor(env, filename=monitor_filename)
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
    parser.add_argument("--n_envs", default=20, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log_dir", default='./logs/')
    parser.add_argument("--experiment_name", default='DQN')
    args = parser.parse_args()

    args.log_dir = args.log_dir + '/' + args.experiment_name
    os.makedirs(args.log_dir, exist_ok=True)
    monitor_filename = os.path.join(args.log_dir, 'monitor.csv')

    def make_env():
        env = make_retro(
            game=args.game,
            state=args.state,
            scenario=args.scenario,
            inttype=retro.data.Integrations.ALL,
            render_mode=args.render_mode,
            monitor_filename=monitor_filename
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
        tensorboard_log="./tensorboard_logs/",
        verbose=1
    )

    if args.checkpoint:
        model.set_parameters(args.checkpoint)

    checkpoint_callback = CheckpointCallback(
        name_prefix=args.experiment_name,
        save_path=args.log_dir,
        save_freq=62500
    )

    record_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))
    video_callback = VideoRecorderCallback(
        record_env,
        save_path=args.log_dir,
        render_freq= 62500,
        n_eval_episodes=1,
        deterministic=True
    )

    # uncomment the following line to turn on GUI while evaluating
    # args.render_mode = 'human'
    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        n_eval_episodes=5,
        eval_freq= 6250,
        deterministic=True,
        render=False,
        verbose=1
    )

    callback = CallbackList([
        eval_callback,
        checkpoint_callback,
        video_callback
    ])

    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        tb_log_name=args.experiment_name,
        reset_num_timesteps=False,
        callback=callback
    )

    model.save(args.log_dir + f"/{args.experiment_name}_last_" + time.strftime("%Y%m%d-%H%M%S"))



if __name__ == "__main__":
    main()
