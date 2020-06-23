import random
from dataclasses import dataclass

from time import time

import numpy as np
from functools import partial
from pprint import pprint
from torch import multiprocessing
from typing import List, NamedTuple, Dict, Any, Callable

from mlutil.crossvalidation import calc_mean_std_scores, calc_scores
from reading_seqtag_data import read_conll03_en, TaggedSeqsDataSet
from seq_tag_util import calc_seqtag_f1_scores
from spacyCrf_score_task import spacycrf_predict_bio
from spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger, Params
from util import data_io
from util.worker_pool import GenericTask


def calc_mean_entropy(probas_seq: List[Dict[str, float]]):
    return np.mean(
        [
            sum([-p * np.log(p) for tag, p in token_probas.items()])
            for token_probas in probas_seq
        ]
    )


def select_by_max_entropy(tagger, data: List, num_to_select: int = 10):

    probas = tagger.predict_marginals(data)
    scores = [calc_mean_entropy(sent) for sent in probas]

    idx = [i for s, i in sorted(zip(scores, range(len(data))), key=lambda x: -x[0])][
        :num_to_select
    ]
    return idx


def select_random(tagger, data: List, num_to_select: int = 10):
    idx = list(range(len(data)))
    random.shuffle(idx)
    return idx[:num_to_select]


class AlTaskData(NamedTuple):
    params: Any
    data: TaggedSeqsDataSet


@dataclass
class Job:
    select_fun: Callable
    scores: Dict = None


class ActiveLearnSpacyCrfSeqTagScoreTask(GenericTask):
    @staticmethod
    def build_task_data(**task_params) -> AlTaskData:
        data: TaggedSeqsDataSet = task_params["data_supplier"]()
        return AlTaskData(params=task_params["params"], data=data)

    @classmethod
    def process(cls, job: Job, task_data: AlTaskData):
        data = task_data.data.train
        train_data_len = len(data)
        step = round(0.01 * train_data_len)
        idx = np.random.randint(0, high=train_data_len, size=(step))
        eval_metrices = []
        chosen_data = []
        for al_step in range(10):

            chosen_data += [data[i] for i in idx]
            data = [d for k, d in enumerate(data) if k not in idx]
            predictions, idx = cls.predict_and_pick_from_corpus(
                partial(job.select_fun, num_to_select=step),
                chosen_data,
                data,
                task_data.data.test,
                task_data.params,
            )
            eval_metrics = {
                split_name: calc_seqtag_f1_scores(preds, targets)
                for split_name, (preds, targets) in predictions.items()
            }
            eval_metrices.append(
                {"train_size": len(chosen_data), "scores": eval_metrics}
            )
        job.scores = eval_metrices
        job.select_fun = job.select_fun.__name__
        return job.__dict__

    @classmethod
    def predict_and_pick_from_corpus(cls, select_fun, train, corpus, test, params):

        tagger = SpacyCrfSuiteTagger(params=params)
        tagger.fit(train)

        predictions = {
            split_name: spacycrf_predict_bio(tagger, split_data)
            for split_name, split_data in {"train": train, "test": test}.items()
        }
        data = [[token for token, tag in datum] for datum in corpus]
        idx = select_fun(tagger, data)
        return predictions, idx


if __name__ == "__main__":
    import os

    data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )
    dataset = data_supplier()

    task = ActiveLearnSpacyCrfSeqTagScoreTask(
        params=Params(c1=0.5, c2=0.0, max_it=100), data_supplier=data_supplier
    )
    num_folds = 5
    select_funs = [select_by_max_entropy, select_random]
    jobs = [Job(f) for _ in range(num_folds) for f in select_funs]
    num_workers = min(multiprocessing.cpu_count() - 1, len(jobs))
    start = time()
    scores = calc_scores(task, jobs, num_workers)
    duration = time() - start
    print(
        "%d jobs with %d workers took: %0.2f seconds"
        % (len(jobs), num_workers, duration)
    )
    data_io.write_jsonl("scores.jsonl", scores)
