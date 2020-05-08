import re
from typing import List, Dict, Tuple, NamedTuple
from flair.data import Sentence, Token, Corpus, Dictionary
from torch.utils.data import Dataset

from seq_tag_util import char_precise_spans_to_token_spans
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


class SciercDocument(NamedTuple):
    sentences: List
    ner: List
    clusters: List = None
    relations: List = None
    doc_key: List = None


def build_tagged_scierc_sequences(
    sentences: List[List[str]], ner: List[List]
) -> List[List[Tuple[str, str]]]:
    def build_tag(index, ner_spans) -> str:
        spans_overlapping_with_index = [
            (start, end, label)
            for start, end, label in ner_spans
            if index >= start and index <= end
        ]
        if len(spans_overlapping_with_index) > 1:
            spans_overlapping_with_index = [spans_overlapping_with_index[0]]# TODO(tilo)!!
            # assert False
        if len(spans_overlapping_with_index) == 1:
            start, end, label = spans_overlapping_with_index[0]
            tag = prefix_to_BIOES(label, start, end, index)
        else:
            tag = "O"
        return tag

    def get_tagged_sequences():
        offset = 0
        for tokens, token_spans in zip(sentences, ner):
            token_spans = [
                s
                for k, s in enumerate(token_spans)
                if not another_span_is_wider(s, k, token_spans)
            ]
            assert all([l in LABELS for _, _, l in token_spans])
            tagged_sequence = [
                (token, build_tag(token_index + offset, token_spans))
                for token_index, token in enumerate(tokens)
            ]
            yield tagged_sequence
            offset += len(tokens)

    return list(get_tagged_sequences())


def read_scierc_seqs(
    jsonl_file, process_fun=lambda x: (x["sentences"], x["ner"])
) -> List[List[Tuple[str, str]]]:
    seqs = [
        sent
        for sentences, ner in (process_fun(d) for d in data_io.read_jsonl(jsonl_file))
        for sent in build_tagged_scierc_sequences(sentences=sentences, ner=ner)
    ]
    return seqs

import re
def regex_tokenizer(text, pattern=r"(?u)\b\w\w+\b")->List[Tuple[int,int,str]]:# pattern stolen from scikit-learn
    return [(m.start(),m.end(),m.group()) for m in re.finditer(pattern, text)]

use_flair_tokenizer=False

def char_to_token_level(d):

    if use_flair_tokenizer:
        flair_sentence = Sentence(d['text'], use_tokenizer=True)
        token_spans = [(tok.start_position,tok.end_position,tok.text) for tok in flair_sentence.tokens]
    else:
        token_spans = regex_tokenizer(d['text'])

    char_spans = d['labels']
    tagged_token_spans = char_precise_spans_to_token_spans(char_spans, [(s,e) for s,e,t in token_spans])
    sentence = [t for s,e,t in token_spans]
    return [sentence],[tagged_token_spans]

if __name__ == "__main__":
    from pathlib import Path

    home = str(Path.home())
    file = home + "/data/current_corrected_annotations.json"
    data = read_scierc_seqs(file, process_fun=char_to_token_level)
    print()