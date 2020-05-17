from pprint import pprint

import os
from collections import OrderedDict
from functools import partial

from torch import multiprocessing

import flair_score_tasks
from data_splitting import shufflesplit_trainset_only_trainsize_range
from experiment_util import Experiment, TRAINONLY
from mlutil.crossvalidation import (
    calc_scores,
    calc_mean_and_std,
    ScoreTask,
)  # TODO(tilo) must be imported before numpy

import numpy
from itertools import groupby
from time import time
from typing import Dict, List, Tuple, Any, Iterable

from reading_seqtag_data import (
    read_scierc_data,
    read_JNLPBA_data,
    TaggedSeqsDataSet,
    read_conll03_en,
)
from spacyCrf_score_task import SpacyCrfScorer
from util import data_io


def groupandsort_by_first(tups: Iterable[Tuple[Any, Any]]):
    by_first = lambda xy: xy[0]
    return OrderedDict(
        [
            (k, [l for _, l in group])
            for k, group in groupby(sorted(tups, key=by_first), key=by_first)
        ]
    )


home = os.environ["HOME"]


def tuple_2_dict(t):
    m, s = t
    return {"mean": m, "std": s}


def calc_write_learning_curve(exp: Experiment, max_num_workers=40):
    num_workers = min(
        min(max_num_workers, multiprocessing.cpu_count() - 1), exp.num_folds
    )

    name = exp.name
    print("got %d evaluations to calculate" % len(exp.jobs))
    results_path = results_folder + "/" + name
    os.makedirs(results_path, exist_ok=True)
    start = time()
    scores = calc_scores(
        exp.score_task, [split for train_size, split in exp.jobs], n_jobs=num_workers
    )
    duration = time() - start
    meta_data = {
        "duration": duration,
        "num-workers": num_workers,
        "experiment": str(exp),
    }
    data_io.write_json(results_path + "/meta_datas.json", meta_data)
    print("calculating learning-curve for %s took %0.2f seconds" % (name, duration))
    pprint(scores)
    results = groupandsort_by_first(
        zip([train_size for train_size, _ in exp.jobs], scores)
    )
    data_io.write_json(results_path + "/learning_curve.json", results)

    trainsize_to_mean_std_scores = {
        train_size: tuple_2_dict(calc_mean_and_std(m))
        for train_size, m in results.items()
    }
    data_io.write_json(
        results_path + "/learning_curve_meanstd.json", trainsize_to_mean_std_scores,
    )


if __name__ == "__main__":
    # data_supplier= partial(read_scierc_data,path=home + "/data/scierc_data/sciERC_processed/processed_data/json")
    # data_supplier = partial(
    #     read_scierc_seqs,
    #     jsonl_file=home + "/data/scierc_data/final_data.json",
    #     process_fun=char_to_token_level,
    # )
    data_path = os.environ["HOME"] + "/scibert/data/ner/JNLPBA"

    def data_supplier():
        data = read_JNLPBA_data(data_path)
        return data._asdict()

    # data_supplier = partial(
    #     read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    # )

    results_folder = home + "/data/seqtag_results/learn_curve_JNLPBA"
    os.makedirs(results_folder, exist_ok=True)

    dataset = data_supplier()

    num_folds = 3
    splits = shufflesplit_trainset_only_trainsize_range(
        TaggedSeqsDataSet(**dataset), num_folds=num_folds, train_sizes=[0.05],
    )
    import farm_score_tasks

    exp = Experiment(
        "farm",
        TRAINONLY,
        num_folds=num_folds,
        jobs=splits,
        score_task=farm_score_tasks.FarmSeqTagScoreTask(
            params=farm_score_tasks.Params(n_epochs=1), data_supplier=data_supplier
        ),
    )
    calc_write_learning_curve(exp, max_num_workers=0)

    exp = Experiment(
        "flair-pooled",
        TRAINONLY,
        num_folds=num_folds,
        jobs=splits,
        score_task=flair_score_tasks.BiLSTMConll03enPooled(
            params=flair_score_tasks.Params(max_epochs=3), data_supplier=data_supplier
        ),
    )
    calc_write_learning_curve(exp, max_num_workers=0)

    exp = Experiment(
        "flair",
        TRAINONLY,
        num_folds=num_folds,
        jobs=splits,
        score_task=flair_score_tasks.BiLSTMConll03en(
            params=flair_score_tasks.Params(max_epochs=3), data_supplier=data_supplier
        ),
    )
    calc_write_learning_curve(exp, max_num_workers=0)

    import spacy_features_sklearn_crfsuite as spacy_crf

    exp = Experiment(
        "spacy-crf",
        TRAINONLY,
        num_folds=num_folds,
        jobs=splits,
        score_task=SpacyCrfScorer(
            params=spacy_crf.Params(max_it=3), data_supplier=data_supplier
        ),
    )
    calc_write_learning_curve(exp, max_num_workers=40)
