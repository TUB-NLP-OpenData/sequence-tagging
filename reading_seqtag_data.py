import os
from typing import Dict, NamedTuple, Tuple, List

from allennlp.data.dataset_readers import Conll2003DatasetReader
from reading_scierc_data import read_scierc_seqs


class TaggedSeqsDataSet(NamedTuple):
    train: List[List[Tuple[str, str]]]
    dev: List[List[Tuple[str, str]]]
    test: List[List[Tuple[str, str]]]


def read_scierc_data(path)->TaggedSeqsDataSet:
    data = {
        dataset_name: read_scierc_seqs("%s/%s.json" % (path, dataset_name))
        for dataset_name in ["train", "dev", "test"]
    }
    return TaggedSeqsDataSet(**data)


def read_JNLPBA_data(path) -> TaggedSeqsDataSet:
    conll_reader = Conll2003DatasetReader()

    def read_file(file):
        instances = conll_reader.read(file)
        return [
            [
                (tok.text, tag)
                for tok, tag in zip(instance.fields["tokens"], instance.fields["tags"])
            ]
            for instance in instances
        ]

    dataset2sequences = {
        file.split(".")[0]: list(read_file("%s/%s" % (path, file)))
        for file in os.listdir(path)
    }
    return TaggedSeqsDataSet(**dataset2sequences)


def get_JNLPBA_sequences(jnlpda_data_path)->List[List[Tuple[str, str]]]:
    dataset: TaggedSeqsDataSet = read_JNLPBA_data(jnlpda_data_path)
    data = [
        sent
        for sequences in [dataset.train, dataset.dev, dataset.test]
        for sent in sequences
    ]
    return data

if __name__ == "__main__":
    path = "../scibert/data/ner/JNLPBA"
    data = read_JNLPBA_data(path)
    print()
