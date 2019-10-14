import pandas as pd

from util import data_io

if __name__ == '__main__':
    import seaborn as sns
    from pathlib import Path

    home = str(Path.home())
    data_path = home + '/data/scierc_data'

    sns.set(style="ticks", palette="pastel")
    file = data_path+'/learning_curve_scores.json'
    d = data_io.read_json(file)
    data = [{'train_size':round(float(train_size),2),'f1-spanwise':score[traintest]['f1-spanwise'],'traintest':traintest} for train_size,scores in d.items() for score in scores for traintest in ['train','test']]
    df = pd.DataFrame(data=data)

    ax = sns.boxplot(x="train_size", y="f1-spanwise",
                     hue="traintest", palette=["m", "g"],
                     data=df)
    # sns.despine(offset=10, trim=True)
    ax.set_title('spacy+crfsuite with 3-fold-crossval')
    ax.figure.savefig("learning_curve.png")