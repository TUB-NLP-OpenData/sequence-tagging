import multiprocessing
from itertools import groupby
from time import time
from typing import Dict, List, Tuple, Any, Iterable
from sklearn.model_selection import ShuffleSplit

from benchmark_flair_tagger import score_flair_tagger
from reading_scierc_data import (
    build_tag_dict,
    TAG_TYPE,
    build_flair_sentences_from_sequences,
)
from reading_seqtag_data import read_scierc_data
from util import data_io

from benchmark_spacyCrf_tagger import score_spacycrfsuite_tagger
from mlutil.crossvalidation import calc_scores, calc_mean_and_std, ScoreTask


def groupbykey(x: List[Dict]):
    return {
        k: [l for _, l in group]
        for k, group in groupby(
            sorted([(k, v) for d in x for k, v in d.items()], key=lambda x: x[0]),
            key=lambda x: x[0],
        )
    }


def groupbyfirst(x: Iterable[Tuple[Any, Any]]):
    return {
        k: [l for _, l in group]
        for k, group in groupby(
            sorted([(k, v) for k, v in x], key=lambda x: x[0]), key=lambda x: x[0]
        )
    }


from pathlib import Path

home = str(Path.home())
data_path = home + "/data/scierc_data/processed_data/json"
results_path = home + "/data/scierc_data"
# data_path = "../scibert/data/ner/JNLPBA"


def load_datasets():
    return read_scierc_data(data_path)


def tuple_2_dict(t):
    m, s = t
    return {"mean": m, "std": s}


def calc_write_learning_curve(
    name, kwargs_builder, scorer_fun, params, splits, n_jobs=0
):
    start = time()
    task = ScoreTask(scorer_fun, kwargs_builder, params)

    scores = calc_scores(task, [split for train_size, split in splits], n_jobs=n_jobs)
    print(
        "calculating learning-curve for %s took %0.2f seconds" % (name, time() - start)
    )
    results = groupbyfirst(zip([train_size for train_size, _ in splits], scores))
    data_io.write_json(results_path + "/learning_curve_%s.json" % name, results)
    trainsize_to_mean_std_scores = {
        train_size: tuple_2_dict(calc_mean_and_std(m))
        for train_size, m in results.items()
    }
    data_io.write_json(
        results_path + "/learning_curve_meanstd_%s.json" % name,
        trainsize_to_mean_std_scores,
    )


def flair_kwargs_builder(params):
    data = load_datasets()

    def train_dev_test_sentences_builder(split, data):
        return [
            build_flair_sentences_from_sequences(
                [getattr(data,dataset_name)[i] for i in split[dataset_name]]
            )
            for dataset_name in ["train", "dev", "test"]
        ]

    return {
        "data": data,
        "params": params,
        "tag_dictionary": build_tag_dict(
            [seq for seqs in data._asdict().values() for seq in seqs], TAG_TYPE
        ),
        "train_dev_test_sentences_builder": train_dev_test_sentences_builder,
    }


def spacyCrfSuite_kwargs_supplier(params):
    data = load_datasets()
    return {
        "data": data,
        "params": params,
        "datasets_builder_fun": lambda split, data: {
            dataset_name: [getattr(data,dataset_name)[i] for i in indizes]
            for dataset_name, indizes in split.items()
        },
    }


if __name__ == "__main__":
    dataset = load_datasets()

    num_folds = 4
    splits = [
        (
            train_size,
            {
                "train": train,
                "dev": list(range(len(dataset.dev))),
                "test": list(range(len(dataset.test))),
            },
        )
        for train_size in [0.99]  # np.arange(0.1,1.0,0.3).tolist()+[0.99]
        for train, _ in ShuffleSplit(
            n_splits=num_folds, train_size=train_size, test_size=None, random_state=111
        ).split(X=range(len(dataset.train)))
    ]
    print("got %d evaluations to calculate" % len(splits))

    calc_write_learning_curve(
        "spacyCrfSuite",
        spacyCrfSuite_kwargs_supplier,
        score_spacycrfsuite_tagger,
        {"params": {"c1": 0.5, "c2": 0.0}},
        splits,
        min(multiprocessing.cpu_count() - 1, len(splits)),
    )

    num_jobs = 2
    calc_write_learning_curve(
        "flair",
        flair_kwargs_builder,
        score_flair_tagger,
        {"params": {"max_epochs": 9}},
        splits,
        num_jobs,
    )
