from abc import abstractmethod
from typing import List, Dict, Any, Callable

from flair.data import Sentence, Token, Corpus

from experiment_util import split_splits, split_data, SeqTagScoreTask
from reading_seqtag_data import TaggedSequence, TaggedSeqsDataSet
from seq_tag_util import bilou2bio, calc_seqtag_f1_scores
from util.worker_pool import GenericTask


def build_flair_sentences_from_sequences(
    sequences: List[TaggedSequence], TAG_TYPE="ner"
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


def build_tag_dict(sequences: List[TaggedSequence], tag_type):
    sentences = build_flair_sentences_from_sequences(sequences)
    corpus = Corpus(train=sentences, dev=[], test=[])
    return corpus.make_tag_dictionary(tag_type)


def build_task_data_maintaining_splits(params, data_supplier):
    dataset: TaggedSeqsDataSet = data_supplier()

    def train_dev_test_sentences_builder(split, data):
        return {
            split_name: build_flair_sentences_from_sequences(data_split)
            for split_name, data_split in split_splits(split, data).items()
        }

    return {
        "data": dataset._asdict(),
        "params": params,
        "tag_dictionary": build_tag_dict(
            [seq for seqs in dataset._asdict().values() for seq in seqs], TAG_TYPE
        ),
        "train_dev_test_sentences_builder": train_dev_test_sentences_builder,
    }


def build_task_data_concat_dataset(params, data_supplier):
    dataset: TaggedSeqsDataSet = data_supplier()
    sentences: List = dataset.train + dataset.dev + dataset.test

    def train_dev_test_sentences_builder(split, data):
        return {
            split_name: build_flair_sentences_from_sequences(data_split)
            for split_name, data_split in split_data(split, data).items()
        }

    return {
        "data": sentences,
        "params": params,
        "tag_dictionary": build_tag_dict(sentences, TAG_TYPE),
        "train_dev_test_sentences_builder": train_dev_test_sentences_builder,
    }


from flair.trainers import trainer
import logging

logger = trainer.log
logger.setLevel(logging.WARNING)

TAG_TYPE = "ner"


class FlairScoreTask(SeqTagScoreTask):
    @staticmethod
    def build_task_data(**task_params) -> Dict[str, Any]:
        return build_task_data_maintaining_splits(**task_params)

    @classmethod
    def predict_with_targets(cls, job, task_data: Dict[str, Any]):
        split = job

        # torch.cuda.empty_cache()

        splits = task_data["train_dev_test_sentences_builder"](split, task_data["data"])

        corpus = Corpus(train=splits["train"], dev=splits["dev"], test=splits["test"])

        tagger = cls.build_train_sequence_tagger(
            corpus, task_data["tag_dictionary"], task_data["params"]
        )

        def flair_tagger_predict_bio(sentences: List[Sentence]):
            train_data = [
                [(token.text, token.tags[tagger.tag_type].value) for token in datum]
                for datum in sentences
            ]
            targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]

            pred_sentences = tagger.predict(sentences)

            SPECIAL_FLAIR_TAGS = ["<START>","<STOP>","<unk>"]

            def replace_start_token_with_O(seq: List[str]):
                return ["O" if t in SPECIAL_FLAIR_TAGS else t for t in seq]

            pred_data = [
                bilou2bio(
                    replace_start_token_with_O(
                        [token.tags[tagger.tag_type].value for token in datum]
                    )
                )
                for datum in pred_sentences
            ]
            return pred_data, targets

        return {n: flair_tagger_predict_bio(d) for n, d in splits.items()}

    @staticmethod
    @abstractmethod
    def build_train_sequence_tagger(corpus, tag_dictionary, params, TAG_TYPE="ner"):
        raise NotImplementedError
