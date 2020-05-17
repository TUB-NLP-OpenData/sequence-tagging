from abc import abstractmethod
from dataclasses import dataclass

from typing import List, Any, Callable, Dict, Tuple, NamedTuple, Union

from data_splitting import split_splits, LearnCurveJob, EvalJob
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

Splits = Dict[str, List[int]]

class SeqTagTaskData(NamedTuple):
    data: Dict[str, List]
    task_data: Any


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
        splits = split_splits(job, task_data.data)
        predictions = cls.predict_with_targets(splits, task_data.task_data)
        return {
            split_name: calc_seqtag_f1_scores(preds, targets)
            for split_name, (preds, targets) in predictions.items()
        }

    @classmethod
    @abstractmethod
    def predict_with_targets(
        cls, splits:Splits, params
    ) -> Dict[str, Tuple[Sequences, Sequences]]:
        raise NotImplementedError
