from sklearn.model_selection import ShuffleSplit
from typing import Dict, List, NamedTuple


EvalJob = Dict[str, List[int]]


class TrainDevTest(NamedTuple):
    train: List
    dev: List
    test: List


def shufflesplit_trainset_only(
    dataset: TrainDevTest, num_folds: int = 5, train_size=0.8
) -> List[EvalJob]:
    splitter = ShuffleSplit(n_splits=num_folds, train_size=train_size, random_state=111)
    splits = [
        {
            "train": train,
            "dev": list(range(len(dataset.dev))),
            "test": list(range(len(dataset.test))),
        }
        for train, _ in splitter.split(X=range(len(dataset.train)))
    ]
    return splits


def crosseval_on_concat_dataset(
    dataset: TrainDevTest, num_folds: int = 5, test_size=0.2
) -> List[EvalJob]:
    sentences = dataset.train + dataset.dev + dataset.test
    splitter = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=111)
    splits = [
        {"train": train, "dev": train[: round(len(train) / 5)], "test": test}
        for train, test in splitter.split(X=range(len(sentences)))
    ]
    return splits


def preserve_train_dev_test(
    dataset: TrainDevTest, num_folds: int=5
) -> List[EvalJob]:
    splits = [
        {
            dsname: list(range(len(getattr(dataset, dsname))))
            for dsname in ["train", "dev", "test"]
        }
    ] * num_folds
    return splits
