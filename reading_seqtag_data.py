import flair.datasets
import os
from farm.data_handler.utils import read_ner_file
from typing import Dict, NamedTuple, Tuple, List

from allennlp.data.dataset_readers import Conll2003DatasetReader
from reading_scierc_data import read_scierc_seqs
from seq_tag_util import iob2iobes, BIOES

TaggedSequence = List[Tuple[str, str]]


class TaggedSeqsDataSet(NamedTuple):
    train: List[TaggedSequence]
    dev: List[TaggedSequence]
    test: List[TaggedSequence]


def preprocess_sequence(seq: TaggedSequence) -> TaggedSequence:
    tags = [tag for tok, tag in seq]
    prepro_tags = iob2iobes(tags)
    assert set([t[0] for t in prepro_tags]).issubset(BIOES), prepro_tags
    return [(tok, tag) for (tok, _), tag in zip(seq, prepro_tags)]


def read_scierc_data(path) -> TaggedSeqsDataSet:
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
        file.split(".")[0]: [
            preprocess_sequence(seq) for seq in read_file("%s/%s" % (path, file))
        ]
        for file in os.listdir(path)
    }
    return TaggedSeqsDataSet(**dataset2sequences)


def read_tokenized_and_tagged(file_path):
    """
    taken from transformers/examples/utils_ner.py
    """

    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append([(w, l) for w, l in zip(words, labels)])
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
            examples.append([(w, l) for w, l in zip(words, labels)])
    return examples


def read_germEval_2014_data(path) -> TaggedSeqsDataSet:
    dataset2sequences = {
        file.split(".")[0]: list(read_tokenized_and_tagged("%s/%s" % (path, file)))
        for file in ["train.txt", "dev.txt", "test.txt"]
    }
    return TaggedSeqsDataSet(**dataset2sequences)


def read_conll03_en(path: str):
    dataset_name = "conll03-en"
    data = dict()
    for dataset_file in ["train.txt", "dev.txt", "test.txt"]:
        file = os.path.join(path, dataset_name, dataset_file)
        data_dict = read_ner_file(file, sep=" ")
        tagged_seqs = [
            [(tok, tag) for tok, tag in zip(d["text"].split(" "), d["ner_label"])]
            for d in data_dict
        ]
        split_name = dataset_file.split(".")[0]
        prepro_tagseqs = [preprocess_sequence(ts) for ts in tagged_seqs]
        data[split_name] = prepro_tagseqs

    return TaggedSeqsDataSet(**data)


def get_UD_English_data():

    corpus = flair.datasets.UD_ENGLISH()
    train_data_flair = corpus.train
    test_data_flair = corpus.test
    print("train-data-len: %d" % len(train_data_flair))
    print("test-data-len: %d" % len(test_data_flair))

    tag_type = "pos"

    def filter_tags(tag):
        return tag  # if tag_counter[tag] > 50 else 'O'

    train_data = [
        [(token.text, filter_tags(token.tags["pos"].value)) for token in datum]
        for datum in train_data_flair
    ]
    test_data = [
        [(token.text, filter_tags(token.tags["pos"].value)) for token in datum]
        for datum in test_data_flair
    ]
    return train_data, test_data, tag_type


if __name__ == "__main__":
    data_path = os.environ["HOME"] + "/data/IE/sequence_tagging_datasets"
    dataset = read_conll03_en(data_path)
    print()
