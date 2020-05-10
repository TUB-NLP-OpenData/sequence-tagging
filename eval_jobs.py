import numpy
from sklearn.model_selection import ShuffleSplit
from typing import Dict, List, NamedTuple, Tuple

from reading_seqtag_data import TaggedSeqsDataSet

EvalJob = Dict[str, List[int]]


def _build_split_devtest_fix(train, dataset: TaggedSeqsDataSet):
    return {
        "train": train,
        "dev": list(range(len(dataset.dev))),
        "test": list(range(len(dataset.test))),
    }


def shufflesplit_trainset_only(
    dataset: TaggedSeqsDataSet, num_folds: int = 5, train_size=0.8
) -> List[EvalJob]:
    splitter = ShuffleSplit(n_splits=num_folds, train_size=train_size, random_state=111)
    splits = [
        _build_split_devtest_fix(train, dataset)
        for train, _ in splitter.split(X=range(len(dataset.train)))
    ]
    return splits


def shufflesplit_trainset_only_trainsize_range(
    dataset: TaggedSeqsDataSet, num_folds=3, train_sizes=[0.1, 0.5, 0.99]
) -> List[Tuple[float, EvalJob]]:
    splits = [
        (train_size, _build_split_devtest_fix(train, dataset))
        for train_size in train_sizes
        for train, _ in ShuffleSplit(
            n_splits=num_folds, train_size=train_size, test_size=None, random_state=111
        ).split(X=range(len(dataset.train)))
    ]
    return splits


def build_train_sizes(starts, ends, steps):
    train_sizes = numpy.arange(starts, ends, steps).tolist() + [0.99]
    return train_sizes


def crosseval_on_concat_dataset(
    dataset: TaggedSeqsDataSet, num_folds: int = 5, test_size=0.2
) -> List[EvalJob]:
    sentences = dataset.train + dataset.dev + dataset.test
    splitter = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=111)
    splits = [
        {"train": train, "dev": train[: round(len(train) / 5)], "test": test}
        for train, test in splitter.split(X=range(len(sentences)))
    ]
    return splits


def _build_split_numtrain(train_proportion, not_test, test):
    num_train = int(round(train_proportion * len(not_test)))
    splits_dict = {
        "train": not_test[:num_train],
        "dev": not_test[:num_train],
        "test": test,
    }
    return splits_dict


def crosseval_on_concat_dataset_trainsize_range(
    dataset_size, num_folds=3, test_size=0.2, starts=0.1, ends=1.0, steps=0.3
) -> List[Tuple[float, EvalJob]]:
    train_sizes = build_train_sizes(starts, ends, steps)

    def calc_proportion_of_overall_dataset(rel_train_size, not_test):
        return rel_train_size * (len(not_test) / dataset_size)

    splits = [
        (
            calc_proportion_of_overall_dataset(rel_train_size, not_test),
            _build_split_numtrain(rel_train_size, not_test, test),
        )
        for rel_train_size in train_sizes
        for not_test, test in ShuffleSplit(
            n_splits=num_folds, test_size=test_size, random_state=111,
        ).split(X=list(range(dataset_size)))
    ]
    return splits


def preserve_train_dev_test(dataset: TaggedSeqsDataSet, num_folds: int = 5) -> List[EvalJob]:
    splits = [
        {
            dsname: list(range(len(getattr(dataset, dsname))))
            for dsname in ["train", "dev", "test"]
        }
    ] * num_folds
    return splits
