import numpy
from sklearn.model_selection import ShuffleSplit
from typing import Dict, List, NamedTuple, Tuple, Any

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


LearnCurveJob = Tuple[float, EvalJob]


def shufflesplit_trainset_only_trainsize_range(
    dataset: TaggedSeqsDataSet, num_folds=3, train_sizes=[0.1, 0.5, 0.99]
) -> List[LearnCurveJob]:
    splits = [
        (train_size, _build_split_devtest_fix(train, dataset))
        for train_size in train_sizes
        for train, _ in ShuffleSplit(
            n_splits=num_folds, train_size=train_size, test_size=None, random_state=111
        ).split(X=range(len(dataset.train)))
    ]
    return splits

def build_data_supplier_splits_trainset_only(
    raw_data_supplier, num_folds, train_size=0.1
):
    def data_supplier():
        data = raw_data_supplier()
        return data._asdict()

    dataset = raw_data_supplier()
    splits = shufflesplit_trainset_only(dataset, num_folds, train_size=train_size)
    return data_supplier, splits


def build_data_supplier_splits_concat(raw_data_supplier, num_folds, test_size=0.1):
    def data_supplier():
        dataset = raw_data_supplier()
        data = dataset.train + dataset.dev + dataset.test
        return {k: data for k in ["train", "dev", "test"]}

    dataset = data_supplier()
    splits = crosseval_on_concat_dataset(
        dataset["train"], num_folds, test_size=test_size
    )
    return data_supplier, splits

def split_splits(split: Dict[str, List[int]], data_splits: Dict[str, List]):
    return {
        split_name: [data_splits[split_name][i] for i in indizes]
        for split_name, indizes in split.items()
    }


def build_train_sizes(starts, ends, steps):
    train_sizes = numpy.arange(starts, ends, steps).tolist() + [0.99]
    return train_sizes


def crosseval_on_concat_dataset(
    data:List, num_folds: int = 5, test_size=0.2
) -> List[EvalJob]:
    splitter = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=111)
    splits = [
        {"train": train, "dev": train[: round(len(train) / 5)], "test": test}
        for train, test in splitter.split(X=range(len(data)))
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
) -> List[LearnCurveJob]:
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
