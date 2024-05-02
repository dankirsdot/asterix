# This file is a modification of the results plotter provided by Stable Baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/results_plotter.py

from typing import Callable, List, Optional, Tuple

import ast
import numpy as np
import pandas as pd
import seaborn as sns

# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt

from stable_baselines3.common.monitor import load_results

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 50


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: the input Array
    :param window: length of the rolling window
    :return: rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = (*array.strides, array.strides[-1])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides).astype(float)


def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: variable 1
    :param var_2: variable 2
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2 (such as np.mean)
    :return:  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    errors = np.std(var_2_window, axis=-1) / np.sqrt(window)
    return var_1[window - 1 :], function_on_var2, errors


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """    
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)        
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_curves(
    xy_list: List[Tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, labels: List[str], figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    sns.set_theme(style='whitegrid')
    sns.set_palette(sns.color_palette("pastel"))
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for i, (x, y) in enumerate(xy_list):                 
                             
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:        
            # Compute and plot rolling mean and errors with window of size EPISODE_WINDOW        
            x, y_mean, errors = window_func(x, y, EPISODES_WINDOW, np.mean)
            lower_bound = y_mean - errors
            upper_bound = y_mean + errors
            sns.lineplot(x=x, y=y_mean, label=labels[i])
            plt.fill_between(x, lower_bound, upper_bound, alpha=0.5) 
        else:
            sns.lineplot(x=x, y=y, errorbar='sd')
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results(
    dirs: List[str], num_timesteps: Optional[int], labels: List[str], x_axis: str, task_name: str, figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, labels, figsize)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs="*")
    parser.add_argument('--num_steps', '-n', default=None)
    parser.add_argument('--labels', '-l', nargs="*")
    parser.add_argument('--x_axis', '-x', choices=[X_TIMESTEPS, X_EPISODES, X_WALLTIME], default=X_TIMESTEPS)
    parser.add_argument('--task_name', '-t', type=str)
    parser.add_argument('--figsize', '-f', type=str, default="(10,6)")
    args = parser.parse_args()

    """
    Args:
    dirs (strings): As many log directories as you'd like to plot from. Do note
        that the directories must contain the `monitor.csv` files.
    num_steps (int): Only plot the points below this value.
    labels (strings): The labels that correspond to each experiment in a log
        directory. 
    x_axis (str): The axis for the x and y output. Can be one of the following:
                 - 'timesteps'
                 - 'episodes'
                 - 'walltime_hrs'
    task_name (str): The title of the task to plot.
    figsize (tuple): Size of the figure, specified as a tuple of width and 
        height (e.g. "(10, 6)"). Do note that quotation marks are required.

    Example Usage:
        python3 plot.py --dirs ./logs/PPO ./logs/DQN -n 1000000 -l PPO DQN -x timesteps -t Rewards -f "(10, 3)"
    """
    num_steps = None
    if args.num_steps:
        num_steps = int(args.num_steps)
    figsize = ast.literal_eval(args.figsize)
    plot_results(args.dirs, num_steps, args.labels, args.x_axis, args.task_name, figsize)

if __name__ == "__main__":
    main()

