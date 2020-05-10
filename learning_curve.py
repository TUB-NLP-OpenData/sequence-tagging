import os
from collections import OrderedDict
from functools import partial

from torch import multiprocessing

from experiment_util import Experiment, TRAINONLY
from mlutil.crossvalidation import (
    calc_scores,
    calc_mean_and_std,
    ScoreTask,
)  # TODO(tilo) must be imported before numpy

from eval_jobs import shufflesplit_trainset_only_trainsize_range
import numpy
from itertools import groupby
from time import time
from typing import Dict, List, Tuple, Any, Iterable

from reading_seqtag_data import read_scierc_data, read_JNLPBA_data, TaggedSeqsDataSet
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


def calc_write_learning_curve(
    exp: Experiment, results_path=home + "/data/seqtag_results", max_num_workers=40
):
    num_workers = min(
        min(max_num_workers, multiprocessing.cpu_count() - 1), exp.num_folds
    )

    name = exp.name
    print("got %d evaluations to calculate" % len(exp.splits))
    results_path = results_path + "/" + name
    os.makedirs(results_path, exist_ok=True)
    start = time()
    scores = calc_scores(
        exp.score_task, [split for train_size, split in exp.splits], n_jobs=num_workers
    )
    duration = time() - start
    meta_data = {
        "duration": duration,
        "num-workers": num_workers,
        "experiment": str(exp),
    }
    data_io.write_json(results_path + "/meta_datas.json", meta_data)
    print("calculating learning-curve for %s took %0.2f seconds" % (name, duration))
    results = groupandsort_by_first(
        zip([train_size for train_size, _ in exp.splits], scores)
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
    # data_path = os.environ["HOME"] + "/code/misc/scibert/data/ner/JNLPBA"
    data_supplier = partial(read_JNLPBA_data, path=data_path)

    dataset: TaggedSeqsDataSet = data_supplier()
    # sentences = dataset.train + dataset.dev + dataset.test
    # dataset_size = len(sentences)

    import benchmark_flair_tagger as flair_seqtag

    num_folds = 1
    splits = shufflesplit_trainset_only_trainsize_range(
        TaggedSeqsDataSet(dataset.train, dataset.dev, dataset.test),
        num_folds=num_folds,
        train_sizes=[0.1],
    )

    exp = Experiment(
        "flair-bert-debug",
        TRAINONLY,
        num_folds=num_folds,
        splits=splits,
        score_task=flair_seqtag.FlairGoveSeqTagScorer(
            params={"max_epochs": 1}, data_supplier=data_supplier
        ),
    )
    calc_write_learning_curve(exp, max_num_workers=0)

    # import benchmark_spacyCrf_tagger as spacy_crf
    #
    # # learn_curve_spacy_crf(
    # #     splits=crosseval_on_concat_dataset_trainsize_range(
    # #         dataset_size, num_folds=3, test_size=0.2, starts=0.2, ends=0.8, steps=0.2
    # #     ),
    # #     build_kwargs_fun=spacy_crf.build_kwargs,
    # # )
    # exp = Experiment(
    #     "spacy-crf-debug",
    #     TRAINONLY,
    #     num_folds=num_folds,
    #     splits=splits,
    #     build_kwargs_fun=spacy_crf.kwargs_builder_maintaining_train_dev_test,
    #     scorer_fun=spacy_crf.score_spacycrfsuite_tagger,
    #     data_supplier=data_supplier,
    #     params={"params": {"c1": 0.5, "c2": 0.0}, "data_supplier": data_supplier},
    # )
    # calc_write_learning_curve(exp, max_num_workers=40)
