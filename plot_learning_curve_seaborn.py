import os
import pandas as pd

from util import data_io


def plot_learncurve(data_path, files, split_name, save_dir="images"):
    def build_method_name(file):
        return file.split("_")[-1].replace(".json", "")

    methods = [build_method_name(f) for f in files]

    sns.set(style="ticks", palette="pastel")
    data = [
        {
            "train_size": round(float(train_size), 2),
            "f1-micro-spanlevel": score[split_name]["f1-micro-spanlevel"],
            "method": build_method_name(file),
        }
        for file in files
        for train_size, scores in data_io.read_json(data_path + "/" + file).items()
        for score in scores
    ]
    num_cross_val = len(data) / len(set([d["train_size"] for d in data])) / len(methods)
    df = pd.DataFrame(data=data)
    ax = sns.boxplot(
        x="train_size", y="f1-micro-spanlevel", hue="method", palette=["m", "g"], data=df
    )
    # sns.despine(offset=10, trim=True)
    ax.set_title(
        "evaluated on %s-set with %d-fold-crossval" % (split_name, num_cross_val)
    )
    ax.set_xlabel("subset of train-dataset in %")
    ax.figure.savefig(
        save_dir + "/learning_curve_%s_%s.png" % (split_name, "-".join(methods))
    )
    from matplotlib import pyplot as plt

    plt.close()


if __name__ == "__main__":
    import seaborn as sns

    home = os.environ['HOME']
    data_path = home + "/data/scierc_data"

    files = ['learning_curve_spacyCrfSuite.json']
    # files = ['learning_curve_flair.json', 'learning_curve_spacyCrfSuite.json']
    # files = ["learning_curve_flair.json"]
    plot_learncurve(data_path, files, "train", save_dir=".")
    plot_learncurve(data_path, files, "test", save_dir=".")
