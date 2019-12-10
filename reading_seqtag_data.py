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

def read_tokenized_and_tagged(file_path):
    '''
    taken from transformers/examples/utils_ner.py
    '''

    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append([(w,l) for w,l in zip(words,labels)])
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append([(w,l) for w,l in zip(words,labels)])
    return examples

def read_germEval_2014_data(path) -> TaggedSeqsDataSet:
    dataset2sequences = {
        file.split(".")[0]: list(read_tokenized_and_tagged("%s/%s" % (path, file)))
        for file in ['train.txt','dev.txt','test.txt']
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
    path = "/home/tilo/data/germEval_2014"
    data = read_germEval_2014_data(path)
    print()
