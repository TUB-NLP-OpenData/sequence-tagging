from abc import abstractmethod
from dataclasses import dataclass

from typing import List, Any, Callable, Dict, Tuple, NamedTuple

from eval_jobs import EvalJob, LearnCurveJob
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
    jobs: List[LearnCurveJob]
    score_task: GenericTask

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if k not in ["jobs"]})


class SeqTagTaskData(NamedTuple):
    split_fun: Callable
    data: Dict[str, List]
    params: Any


class SeqTagScoreTask(GenericTask):
    def __init__(self, params, data_supplier: Callable) -> None:
        task_params = {"params": params, "data_supplier": data_supplier}
        super().__init__(**task_params)

    @staticmethod
    @abstractmethod
    def build_task_data(**task_params) -> SeqTagTaskData:
        raise NotImplementedError

    @classmethod
    def process(cls, job: EvalJob, task_data: SeqTagTaskData):
        splits = task_data.split_fun(job, task_data.data)
        predictions = cls.predict_with_targets(splits, task_data.params)
        return {
            split_name: calc_seqtag_f1_scores(preds, targets)
            for split_name, (preds, targets) in predictions.items()
        }

    @classmethod
    @abstractmethod
    def predict_with_targets(
        cls, splits, params
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
