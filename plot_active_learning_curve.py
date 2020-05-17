from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

from util import data_io


def plot_it(experiments, save_dir="."):
    fig, ax = plt.subplots(figsize=(5, 25))
    sns.set(style="ticks", palette="pastel")
    data = [
        {
            "train_size": step["train_size"],
            "f1-micro-spanlevel": step["scores"]["test"]["seqeval-f1"],
            "select_fun": e["select_fun"],
        }
        for e in experiments
        for step in e["scores"]
    ]
    df = pd.DataFrame(data=data)
    ax = sns.boxplot(
        ax=ax,
        x="train_size",
        y="f1-micro-spanlevel",
        hue="select_fun",
        data=df,
    )
    ax.figure.savefig(save_dir + "/active_learning_curve.png")

    plt.close()


if __name__ == "__main__":

    HOME = os.environ["HOME"]

    # keyfun = lambda x:x['select_fun']
    # data = [(k,[v for v in g]) for k,g in groupby(sorted(s,key=keyfun),key=keyfun)]
    data = data_io.read_jsonl(HOME + "/gunther/sequence-tagging/scores.jsonl")
    plot_it(data)
