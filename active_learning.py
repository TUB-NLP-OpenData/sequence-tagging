import numpy as np
from functools import partial
from pprint import pprint
from typing import List, NamedTuple, Dict, Any

from reading_seqtag_data import read_conll03_en, TaggedSeqsDataSet
from seq_tag_util import calc_seqtag_f1_scores
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
        data = task_data.data.train
        train_data_len = len(data)
        step = round(0.1 * train_data_len)
        scores = np.random.random(size=(train_data_len))
        eval_metrices = []
        chosen_data = []
        for al_step in range(3):
            idx = [
                i for s, i in sorted(zip(scores, range(len(data))), key=lambda x: -x[0])
            ][:step]
            chosen_data += [data[i] for i in idx]
            data = [d for k, d in enumerate(data) if k not in idx]
            predictions, scores = cls.predict_with_targets_and_scores(
                chosen_data, data, task_data.data.test, task_data.params
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
    def predict_with_targets_and_scores(cls, train, corpus, test, params):

        tagger = SpacyCrfSuiteTagger(params=params)
        tagger.fit(train)

        predictions = {
            split_name: spacycrf_predict_bio(tagger, split_data)
            for split_name, split_data in {"train": train, "test": test}.items()
        }

        probas = tagger.predict_marginals(
            [[token for token, tag in datum] for datum in corpus]
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

    task = ActiveLearnSpacyCrfSeqTagScoreTask(
        params=Params(c1=0.5, c2=0.0, max_it=100), data_supplier=data_supplier
    )
    num_workers = 0  # min(multiprocessing.cpu_count() - 1, num_folds)
    with task as t:
        scores = t(job=None)

    pprint(scores)
