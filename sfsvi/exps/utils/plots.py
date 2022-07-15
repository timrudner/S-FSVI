"""Utilities for plotting data."""
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sfsvi.exps.utils.load_utils import distinct_sub_specs
from sfsvi.exps.utils.load_utils import filter_exps
from sfsvi.exps.utils.load_utils import find_diff_keys
from sfsvi.exps.utils.load_utils import get_average_test_accuracy_per_task
from sfsvi.exps.utils.load_utils import get_final_avg_valid_accuracy
from sfsvi.exps.utils.load_utils import get_make_immutable_config
from sfsvi.exps.utils.load_utils import load_raw_training_log
from sfsvi.exps.utils.load_utils import load_up_to_date_training_df
from sfsvi.exps.utils.load_utils import rank_configs


def get_final_avg_accuracy(training_log_df: pd.DataFrame) -> float:
    dummy_exp = {"chkpt": {"training_log_df": training_log_df}}
    return get_final_avg_valid_accuracy(dummy_exp)


def plot_final_accuracy(exps: List[Dict], key: Union[List[str], str], figsize=(5, 3)):
    keys = key if isinstance(key, list) else [key]
    assert sorted(find_diff_keys(exps)) == sorted(keys), (
        f"There are other differing keys! "
        f"find_diff_keys(exps) = {find_diff_keys(exps)}"
        f"keys = {sorted(keys)}"
    )
    make_immutable_config = get_make_immutable_config(keys)
    series = pd.Series(
        {
            make_immutable_config(e): get_final_avg_accuracy(
                e["chkpt"]["training_log_df"]
            )
            for e in exps
        }
    )
    series.sort_index(inplace=True)
    return plot_final_avg_accuracy(series, xlabels=keys, figsize=figsize)


def plot_final_avg_accuracy(series, xlabels, figsize=None, ratio=0.001):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    series.plot.bar(ax=ax)
    ax.set_ylabel("Final average accuracy")
    ax.set_xlabel(xlabels)
    ax.set_title("Final average accuracy")
    ax.set_ylim(get_ylim(series.values, ratio=ratio))
    ax.grid()
    fig.tight_layout()
    return ax


def get_ylim(values, ratio):
    return min(values) * (1 - ratio), max(values) * (1 + ratio)


def plot_coreset(
    df: pd.DataFrame,
    plot_type="box",
    bar_ci="sd",
    margin_ratio=0.001,
    x="n_inducing_inputs",
    y="final_avg_accuracy",
    hue="coreset",
    hue_order=None,
    title='',
    figsize=None,
    order=None,
    ax=None,
):
    for c in [x, y, hue]:
        if c and c not in df:
            print(
                f"WARNING: {c} is not in df, not enough data to plot\n"
                f"df.info = {df.info()}"
            )
            return
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    # hue_order = ["random", "entropy", "kl"]
    if plot_type == "bar":
        ax = sns.barplot(
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            data=df,
            ax=ax,
            ci=bar_ci,
        )
        title += f"\nerror bar: {bar_ci}"
    elif plot_type == "box":
        ax = sns.boxplot(
            x=x, y=y, hue=hue, hue_order=hue_order, data=df, ax=ax,
        )
    elif plot_type == "swarm":
        ax = sns.swarmplot(
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            order=order,
            data=df,
            ax=ax,
        )
    elif plot_type == "violin":
        ax = sns.violinplot(
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            order=order,
            data=df,
            ax=ax,
            # cut=0.0
        )
    else:
        raise ValueError(plot_type)
    if title is None:
        title = "Final average accuracy of coreset and number of inducing points"
    ax.set_title(title)
    ax.set_ylim(get_ylim(df[y], margin_ratio))
    # ax.grid()
    return ax


def plot_kl(df: pd.DataFrame, margin_ratio=0.001):
    plt.figure()
    ax = plt.gca()
    ax = sns.boxplot(
        x="coreset_kl_heuristic",
        y="final_avg_accuracy",
        # hue_order=hue_order,
        data=df,
        ax=ax,
    )
    ax.set_title("Final average accuracy of different coreset kl heuristic")
    ax.set_ylim(get_ylim(df["final_avg_accuracy"], margin_ratio))
    ax.grid()
    return ax


def loss_info_to_df(batch_log):
    epochs = [k for k in batch_log if "loss_info_epoch" in k]
    dfs = []
    for i in range(len(epochs)):
        df = pd.DataFrame(batch_log[f"loss_info_epoch_{i}"])
        df = df.applymap(lambda x: x.item())
        df["step_in_epoch"] = df.index
        df["epoch"] = i
        dfs.append(df)
    cat_df = pd.concat(dfs)
    return cat_df.set_index(["epoch", "step_in_epoch"])


def plot_task_train_df(df, axes=None, task="", color=None, alpha=1.0):
    keys = df.columns
    if not axes:
        _, axes = plt.subplots(nrows=len(keys) // 2, ncols=2)
        axes = flatten_axes(axes)
    for ax, k in zip(axes, keys):
        ax.plot(df[k].values, label=f"task {task}", color=color, alpha=alpha)
        ax.set_xlabel("global step")
        ax.grid()
        ax.set_title(k)
    plt.suptitle(f"Training dynamics for task {task}")
    plt.tight_layout()
    plt.subplots_adjust()
    return axes


def flatten_axes(axes) -> List:
    flattened = []
    for a in axes:
        if isinstance(a, np.ndarray) or isinstance(a, Sequence):
            flattened.extend(flatten_axes(a))
        else:
            flattened.append(a)
    return flattened


def plot_cl_train_dynamics(
    exp, nrows=2, ncols=2, figsize=(9, 6), steps_range: Tuple[int] = None
):
    log = load_raw_training_log(exp)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = flatten_axes(axes)

    cmap = plt.get_cmap("winter")
    tasks = sorted(log.keys())
    colors = np.linspace(0, 1, len(tasks))
    for i, t in enumerate(tasks):
        df = loss_info_to_df(log[t]["batch_log"])
        if steps_range:
            df = df.iloc[steps_range[0] : steps_range[1]]
        plot_task_train_df(df, axes=axes, task=str(t), color=cmap(colors[i]), alpha=1.0)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
    )
    plt.tight_layout()
    plt.subplots_adjust()
    plt.suptitle(
        f"Training dynamics for {len(tasks)} tasks in {exp['config']['data_training']}"
    )


def loop_over_setting(exps: List[Dict], keys_across_plot: List[str], dashboard: Callable):
    subspecs = distinct_sub_specs(exps, keys=keys_across_plot)
    subspecs = rank_configs(
        subspecs,
        key="data_training",
        value_order=[
            "continual_learning_smnist",
            "continual_learning_pmnist",
            "continual_learning_smnist_sh",
            "continual_learning_sfashionmnist",
        ],
    )
    for sub_spec in subspecs:
        print("-" * 100)
        print("sub_spec: ", sub_spec)
        sub_exps = filter_exps(exps, d=sub_spec)
        dashboard(sub_exps, sub_spec)


def panel_plot(exps: List[Dict], keys_across_plot: List[str], dashboard: Callable):
    """
    Remove legend
    https://stackoverflow.com/questions/54781243/hide-legend-from-seaborn-pairplot

    Single legend
    https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib

    how to adjust for suptitle
    https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle

    adjust the plot positions
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    """
    subspecs = distinct_sub_specs(exps, keys=keys_across_plot)
    subspecs = rank_configs(
        subspecs,
        key="data_training",
        value_order=[
            "continual_learning_smnist",
            "continual_learning_pmnist",
            "continual_learning_smnist_sh",
            "continual_learning_sfashionmnist",
        ],
    )
    fig, axes = plt.subplots(nrows=1, ncols=len(subspecs),
                             figsize=(9, 4), sharey=False)
    for i, sub_spec in enumerate(subspecs):
        ax = axes[i]
        sub_exps = filter_exps(exps, d=sub_spec)
        dashboard(sub_exps, sub_spec, ax=ax)
        ax.legend([], [], frameon=False)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(False)

    fig.text(0.5, 0.04, 'common X', ha='center')
    fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

    lines_labels = [axes[-1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(0.9, 0.25))
    plt.suptitle("This is a title")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.tight_layout(rect=[0.04, 0.23, 1, 0.95])


def plot_dynamics(exp: Dict, task_id: int, accu_task_id: int):
    plt.figure()
    data = exp['chkpt']['training_log_df'].query(f'task_id == {task_id}')[f'test_acc_task{accu_task_id}']
    plt.plot(data)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title(f"Evolution of test accuracy of {accu_task_id}th task during training the {task_id}th task", fontsize=10)
    return plt.gca()


def plot_dynamics_all(exp: Dict):
    plt.figure()
    ax = plt.gca()
    df = exp['chkpt']['training_log_df']
    df.index = pd.Index(list(range(df.shape[0])))
    acc_cols = sorted([c for c in df.columns if "_acc_task" in c])
    df[acc_cols].plot(ax=ax)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Evolution of accuracy of different tasks", fontsize=10)
    plt.legend()
    plt.tight_layout()
    return ax


def plot_training_dynamics_from_raw_log(exp: Dict, task_id: int, accu_task_id: int):
    """
    task_ids are 1-based
    """
    plt.figure()
    df = load_up_to_date_training_df(exp)
    data = df.query(f'task_id == {task_id}')[f'test_acc_task{accu_task_id}']
    plt.plot(data, '-o')
    plt.xlabel("epoch")
    plt.ylabel("training accuracy")
    plt.grid()
    plt.title(f"Evolution of training accuracy of {accu_task_id}th task "
              f"during training the {task_id}th task", fontsize=10)
    return plt.gca()


def plot_vcl(data, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    for k, matrix in data.items():
        v = np.nanmean(matrix, axis=1)
        ax.plot(v, '-o', label=k)
    plt.legend()
    ax.grid()
    return ax


def plot_exp(exp, label, ax=None,):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    accus = get_average_test_accuracy_per_task(exp)
    ax.plot(accus, '-o', label=label)
    ax.legend()
    return ax


def set_rc_params_to_default():
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def hide_top_and_right_spine(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
