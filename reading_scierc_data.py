from typing import List, Dict, Tuple
from flair.data import Sentence, Token, Corpus, Dictionary
from torch.utils.data import Dataset
from util import data_io

TAG_TYPE = "ner"
LABELS = ["Generic", "Task", "Method", "Material", "Metric", "OtherScientificTerm"]


def prefix_to_BIOES(label, start, end, current_index):
    if end - start > 0:
        if current_index == start:
            prefix = "B"
        elif current_index == end:
            prefix = "E"
        else:
            prefix = "I"
    else:
        prefix = "S"

    return prefix + "-" + label


def build_flair_sentences(d: Dict) -> List[Sentence]:
    def tag_it(token: Token, index, ner_spans):
        labels = [
            (start, end, label)
            for start, end, label in ner_spans
            if index >= start and index <= end
        ]

        if len(labels) > 0:
            for start, end, label in labels:
                token.add_tag(TAG_TYPE, prefix_to_BIOES(label, start, end, index))
        else:
            token.add_tag(TAG_TYPE, "O")

    offset = 0
    sentences = []
    for tokens, ner_spans in zip(d["sentences"], d["ner"]):
        assert all([l in LABELS for _, _, l in ner_spans])
        sentence: Sentence = Sentence()
        [sentence.add_token(Token(tok)) for tok in tokens]
        [tag_it(token, k + offset, ner_spans) for k, token in enumerate(sentence)]
        offset += len(tokens)
        sentences.append(sentence)

    return sentences


def read_scierc_file_to_FlairSentences(jsonl_file: str) -> Dataset:
    dataset: Dataset = [
        sent
        for d in data_io.read_jsonl(jsonl_file)
        for sent in build_flair_sentences(d)
    ]
    return dataset


def get_scierc_data_as_flair_sentences(data_path):

    sentences = [
        sent
        for jsonl_file in ["train.json", "dev.json", "test.json"]
        for sent in read_scierc_file_to_FlairSentences(
            "%s/%s" % (data_path, jsonl_file)
        )
    ]
    return sentences


def build_tag_dict(sequences: List[List[Tuple[str, str]]], tag_type):
    sentences = build_flair_sentences_from_sequences(sequences)
    corpus = Corpus(train=sentences, dev=[], test=[], name="scierc")

    # Make the tag dictionary
    tag_dictionary: Dictionary = Dictionary()
    # tag_dictionary.add_item("O")
    for sentence in corpus.get_all_sentences():
        for token in sentence.tokens:
            tag_dictionary.add_item(token.get_tag(tag_type).value)
    # tag_dictionary.add_item("<START>")
    # tag_dictionary.add_item("<STOP>")
    return corpus.make_tag_dictionary(tag_type)


def build_flair_sentences_from_sequences(
    sequences: List[List[Tuple[str, str]]]
) -> List[Sentence]:

    sentences = []
    for seq in sequences:
        sentence: Sentence = Sentence()
        [sentence.add_token(Token(tok)) for tok, tag in seq]
        [
            flair_token.add_tag(TAG_TYPE, tag)
            for (token, tag), flair_token in zip(seq, sentence)
        ]
        sentences.append(sentence)
    return sentences


def another_span_is_wider(s, k, spans):
    return any(
        [(s[0] >= o[0]) and (s[1] <= o[1]) and k != i for i, o in enumerate(spans)]
    )


def build_sequences(d: dict):
    def build_tag(index, ner_spans) -> str:
        spans_overlapping_with_index = [
            (start, end, label)
            for start, end, label in ner_spans
            if index >= start and index <= end
        ]
        assert len(spans_overlapping_with_index) <= 1
        if len(spans_overlapping_with_index) == 1:
            start, end, label = spans_overlapping_with_index[0]
            tag = prefix_to_BIOES(label, start, end, index)
        else:
            tag = "O"
        return tag

    tagged_seqs = []
    offset = 0
    for tokens, token_spans in zip(d["sentences"], d["ner"]):
        token_spans = [
            s
            for k, s in enumerate(token_spans)
            if not another_span_is_wider(s, k, token_spans)
        ]
        assert all([l in LABELS for _, _, l in token_spans])
        tagged_seqs.append(
            [
                (token, build_tag(token_index + offset, token_spans))
                for token_index, token in enumerate(tokens)
            ]
        )
        offset += len(tokens)
    return tagged_seqs


def read_scierc_seqs(jsonl_file):
    seqs = [sent for d in data_io.read_jsonl(jsonl_file) for sent in build_sequences(d)]
    return seqs


if __name__ == "__main__":
    from pathlib import Path

    home = str(Path.home())
    file = home + "/data/scierc_data/processed_data/json/dev.json"
    read_scierc_seqs(file)
