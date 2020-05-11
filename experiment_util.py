from abc import abstractmethod
from dataclasses import dataclass

from typing import List, Any, Callable, Dict, Tuple

from eval_jobs import EvalJob
from reading_seqtag_data import TaggedSequence
from seq_tag_util import calc_seqtag_f1_scores, Sequences
from util.worker_pool import GenericTask

CROSSVALIDATION = "crossvalidation"
TRAINONLY = "trainonly"


@dataclass
class Experiment:
    name: str
    mode: str
    num_folds: int
    jobs: List[EvalJob]
    score_task: GenericTask

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if k not in ["splits"]})


class SeqTagScoreTask(GenericTask):
    def __init__(self, params: Dict, data_supplier: Callable) -> None:
        task_params = {"params": params, "data_supplier": data_supplier}
        super().__init__(**task_params)

    @classmethod
    def process(cls, job:EvalJob, task_data: Dict[str, Any]):
        predictions = cls.predict_with_targets(job,task_data)
        return {
            split_name: calc_seqtag_f1_scores(preds,targets)
            for split_name, (preds,targets) in predictions.items()
        }

    @classmethod
    @abstractmethod
    def predict_with_targets(
        cls, job:EvalJob, task_data: Dict[str, Any]
    ) -> Dict[str, Tuple[Sequences, Sequences]]:
        raise NotImplementedError


def split_data(split: Dict[str, List[int]], data: List[Any]):
    return {
        split_name: [data[i] for i in indizes] for split_name, indizes in split.items()
    }


def split_splits(split: Dict[str, List[int]], data_splits: Dict[str, List[Any]]):
    return {
        split_name: [data_splits[split_name][i] for i in indizes]
        for split_name, indizes in split.items()
    }
