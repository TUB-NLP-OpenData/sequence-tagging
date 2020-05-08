from dataclasses import dataclass

from typing import List, Any, Callable, Dict

CROSSVALIDATION = "crossvalidation"
TRAINONLY = "trainonly"


@dataclass
class Experiment:
    name: str
    mode: str
    num_folds: int
    splits: List[Any]
    build_kwargs_fun: Callable
    scorer_fun: Callable
    data_supplier: Callable
    params: Dict[str, Any]

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if k not in ["splits"]})


def split_data(split: Dict[str, List[int]], data: List[Any]):
    return {
        split_name: [data[i] for i in indizes] for split_name, indizes in split.items()
    }


def split_splits(split: Dict[str, List[int]], data_splits: Dict[str, List[Any]]):
    return {
        split_name: [data_splits[split_name][i] for i in indizes]
        for split_name, indizes in split.items()
    }
