from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

from util import data_io


def plot_it(experiments, save_dir="."):
    fig, ax = plt.subplots(figsize=(6, 12))
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
        ax=ax, x="train_size", y="f1-micro-spanlevel", hue="select_fun", data=df,
    )

    df1 = df[df.train_size == df.train_size[0]]
    num_cross_val = len(
        df1[df1.select_fun == df1.select_fun[0]]
    )  # well I rarely use pandas!

    ax.set_title("conll03-en %s-set scores with %d-fold-crossval" % ("test", num_cross_val))

    ax.figure.savefig(save_dir + "/active_learning_curve.png")

    plt.close()


if __name__ == "__main__":

    data = data_io.read_jsonl("active_learning/results/conll03_en/scores.jsonl")
    plot_it(data,save_dir='active_learning/results/conll03_en')
