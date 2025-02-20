"""
This module contains utility functions for the Binoculars package.

The `convert_to_pandas` function converts human and machine scores to a pandas DataFrame.
The `save_json` function saves experiment details to a JSON file.
The `save_experiment` function saves the experiment results, including performance metrics and plots.
"""

import json
import os
import datetime

import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

COLOR = "black"

mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR
mpl.rcParams["figure.dpi"] = 200
sns.set(style="darkgrid")


def convert_to_pandas(human_scores, machine_scores):
    """
    Converts human and machine scores to a pandas DataFrame.

    Args:
        human_scores (list[float]): The scores for human-generated text.
        machine_scores (list[float]): The scores for machine-generated text.

    Returns:
        pd.DataFrame: A DataFrame containing the scores and their corresponding classes (0 for human, 1 for machine).
    """
    human_scores = human_scores["score"]
    machine_scores = machine_scores["score"]

    df = pd.DataFrame(
        {"score": human_scores + machine_scores, "class": [0] * len(human_scores) + [1] * len(machine_scores)}
    )
    return df


def save_json(data, save_path):
    """
    Saves experiment details to a JSON file.

    Args:
        data (argparse.Namespace): The experiment details.
        save_path (str): The path to save the JSON file.
    """
    data.end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(save_path, "experiments_details.json"), "w", encoding="utf-8") as f:
        json.dump(data.__dict__, f, ensure_ascii=False, indent=4)


def save_experiment(args, score_df, fpr, tpr, f1_score, roc_auc, tpr_at_fpr_0_01):
    """
    Saves the experiment results, including performance metrics and plots.

    Args:
        args (argparse.Namespace): The experiment arguments.
        score_df (pd.DataFrame): The DataFrame containing the scores and predictions.
        fpr (np.ndarray): The false positive rates.
        tpr (np.ndarray): The true positive rates.
        f1_score (float): The F1 score.
        roc_auc (float): The area under the ROC curve.
        tpr_at_fpr_0_01 (float): The true positive rate at 0.01% false positive rate.
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")

    annotation = f"ROC AUC: {roc_auc:.4f}\nF1 Score: {f1_score:.2f}\nTPR at 0.01% FPR:{100 * tpr_at_fpr_0_01:.2f}%"
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=annotation)
    display.plot(ax=ax, linestyle="--")
    ax.set_title(f"{args.dataset_name} (n={len(score_df)})\nMachine Text from {args.machine_text_source}")

    fig.savefig(f"{args.experiment_path}/performance.png", bbox_inches='tight')
    score_df.to_csv(f"{args.experiment_path}/score_df.csv", index=False)
    save_json(args, args.experiment_path)
