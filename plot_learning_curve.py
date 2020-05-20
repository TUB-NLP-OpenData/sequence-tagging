import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from util import data_io


def plot_learncurve(paths, split_name, save_dir="images"):
    def build_method_name(path):
        return path.split("/")[-1]

    methods = [build_method_name(f) for f in paths]
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.set(style="ticks", palette="pastel")
    data = [
        {
            "train_size": 100*round(float(train_size), 2),
            "f1-micro-spanlevel": score[split_name]["f1-micro-spanlevel"],
            "method": build_method_name(path),
        }
        for path in paths
        for train_size, scores in data_io.read_json(
            path + "/learning_curve.json"
        ).items()
        for score in scores
    ]
    num_cross_val = len(data) / len(set([d["train_size"] for d in data])) / len(methods)
    df = pd.DataFrame(data=data)
    ax = sns.boxplot(
        ax=ax,
        x="train_size",
        y="f1-micro-spanlevel",
        hue="method",
        data=df,
    )
    # sns.despine(offset=10, trim=True)
    ax.set_title(
        "evaluated on %s-set with %d-fold-crossval" % (split_name, num_cross_val)
    )
    ax.set_xlabel("subset of train-dataset in %")
    plt.tight_layout()
    ax.figure.savefig(
        save_dir + "/learning_curve_%s_%s.png" % (split_name, "-".join(methods))
    )

    plt.close()


if __name__ == "__main__":

    home = os.environ["HOME"]
    # data_path = home + "/hpc/data/seqtag_results/JNLPBA_20percent"
    data_path = "results/JNLPBA_20percent"
    result_paths = [
        data_path + "/" + p
        for p in os.listdir(data_path)
        if os.path.isdir(data_path + "/" + p)
    ]

    plot_learncurve(result_paths, "train", save_dir=data_path)
    plot_learncurve(result_paths, "test", save_dir=data_path)
