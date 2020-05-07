import os
from collections import OrderedDict
from functools import partial
from mlutil.crossvalidation import calc_scores, calc_mean_and_std, ScoreTask #TODO(tilo) must be imported before numpy

from eval_jobs import crosseval_on_concat_dataset_trainsize_range
import numpy
from itertools import groupby
from time import time
from typing import Dict, List, Tuple, Any, Iterable
from sklearn.model_selection import ShuffleSplit

from benchmark_flair_tagger import score_flair_tagger, \
    kwargs_builder_maintaining_train_dev_test, kwargs_builder
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
    name,
    kwargs_builder,
    scorer_fun,
    params,
    splits,
    n_jobs=0,
    results_path=home + "/data/seqtag_results",
):
    os.makedirs(results_path,exist_ok=True)
    start = time()
    task = ScoreTask(scorer_fun, kwargs_builder, params)

    scores = calc_scores(task, [split for train_size, split in splits], n_jobs=n_jobs)
    print(
        "calculating learning-curve for %s took %0.2f seconds" % (name, time() - start)
    )
    results = groupandsort_by_first(
        zip([train_size for train_size, _ in splits], scores)
    )
    data_io.write_json(results_path + "/learning_curve_%s.json" % name, results)

    trainsize_to_mean_std_scores = {
        train_size: tuple_2_dict(calc_mean_and_std(m))
        for train_size, m in results.items()
    }
    data_io.write_json(
        results_path + "/learning_curve_meanstd_%s.json" % name,
        trainsize_to_mean_std_scores,
    )



def spacyCrfSuite_kwargs_supplier(params, data_supplier):
    data: TaggedSeqsDataSet = data_supplier()
    return {
        "data": data,
        "params": params,
        "datasets_builder_fun": lambda split, data: {
            dataset_name: [getattr(data, dataset_name)[i] for i in indizes]
            for dataset_name, indizes in split.items()
        },
    }


def spacyCrfSuite_kwargs_supplier_single_set(params, data_supplier):
    data: List = data_supplier()
    return {
        "data": data,
        "params": params,
        "datasets_builder_fun": lambda split, data: {
            dataset_name: [data[i] for i in indizes]
            for dataset_name, indizes in split.items()
        },
    }



if __name__ == "__main__":
    # data_supplier= partial(read_scierc_data,path=home + "/data/scierc_data/sciERC_processed/processed_data/json")
    # data_supplier = partial(
    #     read_scierc_seqs,
    #     jsonl_file=home + "/data/scierc_data/final_data.json",
    #     process_fun=char_to_token_level,
    # )
    data_path = os.environ["HOME"] + "/scibert/data/ner/JNLPBA"
    # data_path = os.environ["HOME"] + "/code/misc/scibert/data/ner/JNLPBA"
    data_supplier = partial(read_JNLPBA_data,path=data_path)

    dataset:TaggedSeqsDataSet = data_supplier()
    sentences = dataset.train + dataset.dev + dataset.test
    dataset_size = len(sentences)


    num_folds = 3
    # splits = build_splits(dataset,num_folds)
    splits = crosseval_on_concat_dataset_trainsize_range(dataset_size, num_folds=2,test_size=0.2,starts=0.05,ends=0.1,steps=0.05)
    print("got %d evaluations to calculate" % len(splits))

    calc_write_learning_curve(
        "flair",
        kwargs_builder,
        score_flair_tagger,
        {"params": {"max_epochs": 2},'data_supplier':data_supplier},
        splits,
        2,
    )
