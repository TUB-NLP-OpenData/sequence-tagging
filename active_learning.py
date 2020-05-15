from pprint import pprint

from time import time

from functools import partial
from random import random

from typing import List, NamedTuple, Dict, Any, Tuple

from abc import abstractmethod

from eval_jobs import preserve_train_dev_test
from experiment_util import SeqTagScoreTask, SeqTagTaskData
import numpy as np

from reading_seqtag_data import read_conll03_en, TaggedSeqsDataSet
from seq_tag_util import calc_seqtag_f1_scores, Sequences
from spacyCrf_score_task import spacycrf_predict_bio
from spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger, Params
from util.worker_pool import GenericTask


def calc_mean_entropy(probas_seq: List[Dict]):
    return np.mean(
        [
            sum([-p * np.log(p) for p in token_probas.values()])
            for token_probas in probas_seq
        ]
    )


class AlTaskData(NamedTuple):
    params: Any
    data: TaggedSeqsDataSet


class ActiveLearnSpacyCrfSeqTagScoreTask(GenericTask):
    @staticmethod
    def build_task_data(**task_params) -> AlTaskData:
        data: TaggedSeqsDataSet = task_params["data_supplier"]()
        return AlTaskData(params=task_params["params"], data=data)

    @classmethod
    def process(cls, job, task_data: AlTaskData):
        splits = task_data.data._asdict()
        data = task_data.data.train
        step = round(0.1 * len(data))
        scores = np.random.random(size=(len(data)))
        eval_metrices = []
        for k in range(3):
            chosen_scores_idx = sorted(
                zip(scores, range(len(data))), key=lambda x: -x[0]
            )[:step]
            chosen_data = [data.pop(i) for s, i in chosen_scores_idx]
            splits["train"] = chosen_data
            splits["corpus"] = data
            predictions, scores = cls.predict_with_targets_and_scores(
                splits, task_data.params
            )
            eval_metrics = {
                split_name: calc_seqtag_f1_scores(preds, targets)
                for split_name, (preds, targets) in predictions.items()
            }
            eval_metrices.append(
                {"train_size": len(chosen_data), "scores": eval_metrics}
            )
        return eval_metrices

    @classmethod
    def predict_with_targets_and_scores(cls, splits: Dict[str, List], params):

        tagger = SpacyCrfSuiteTagger(params=params)
        tagger.fit(splits["train"])

        predictions = {
            split_name: spacycrf_predict_bio(tagger, split_data)
            for split_name, split_data in splits.items()
        }

        probas = tagger.predict_marginals(
            [[token for token, tag in datum] for datum in splits["corpus"]]
        )
        scores = [calc_mean_entropy(sent) for sent in probas]

        return predictions, scores


if __name__ == "__main__":
    import os

    data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )
    dataset = data_supplier()
    num_folds = 1
    splits = preserve_train_dev_test(dataset, num_folds)

    task = ActiveLearnSpacyCrfSeqTagScoreTask(
        params=Params(c1=0.5, c2=0.0), data_supplier=data_supplier
    )
    num_workers = 0  # min(multiprocessing.cpu_count() - 1, num_folds)
    with task as t:
        scores = t(job=None)

    pprint(scores)
