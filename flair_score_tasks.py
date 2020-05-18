import torch
from abc import abstractmethod
from flair.data import Sentence, Token, Corpus
from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from functools import partial
from pprint import pprint
from time import time
from typing import List, Dict, Any
from typing import NamedTuple

from data_splitting import build_data_supplier_splits_trainset_only
from experiment_util import SeqTagScoreTask, SeqTagTaskData
from flair_conll2003_en import build_and_train_conll03en_flair_sequence_tagger
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import TaggedSequence
from reading_seqtag_data import read_conll03_en
from seq_tag_util import bilou2bio


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


from flair.trainers import trainer
import logging

logger = trainer.log
logger.setLevel(logging.WARNING)

TAG_TYPE = "ner"


class Params(NamedTuple):
    max_epochs: int = 3


class FlairScoreTask(SeqTagScoreTask):
    @staticmethod
    def build_task_data(params: Params, data_supplier) -> SeqTagTaskData:

        dataset = data_supplier()

        task_data = {
            "params": params,
            "tag_dictionary": build_tag_dict(
                [seq for seqs in dataset.values() for seq in seqs], TAG_TYPE
            ),
        }

        return SeqTagTaskData(data=dataset, task_data=task_data)

    @classmethod
    def predict_with_targets(cls, raw_splits, task_data: Dict[str, Any]):

        # torch.cuda.empty_cache()

        splits = {
            split_name: build_flair_sentences_from_sequences(data_split)
            for split_name, data_split in raw_splits.items()
        }

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

            SPECIAL_FLAIR_TAGS = ["<START>", "<STOP>", "<unk>"]

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


class FlairGoveSeqTagScorer(FlairScoreTask):
    @staticmethod
    def build_train_sequence_tagger(
        corpus, tag_dictionary, params: Params, TAG_TYPE="ner"
    ):
        embeddings = StackedEmbeddings(
            embeddings=[
                WordEmbeddings("glove"),
                # BertEmbeddings("bert-base-cased", layers="-1")
            ]
        )
        tagger: SequenceTagger = SequenceTagger(
            hidden_size=64,  # 200 with Bert; 64 with glove
            rnn_layers=1,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=TAG_TYPE,
            locked_dropout=0.01,
            dropout=0.01,
            use_crf=False,
        )
        trainer: ModelTrainer = ModelTrainer(
            tagger, corpus, optimizer=torch.optim.Adam, use_tensorboard=False,
        )
        # print(tagger)
        # pprint([p_name for p_name, p in tagger.named_parameters()])

        # process_name = multiprocessing.current_process().name
        # save_path = "flair_seq_tag_model_%s" % process_name
        # if os.path.isdir(save_path):
        #     shutil.rmtree(save_path)
        # assert not os.path.isdir(save_path)
        trainer.train(
            base_path="flair_checkpoints",
            learning_rate=0.001,
            mini_batch_size=128,  # 6 with Bert, 128 with glove
            max_epochs=params.max_epochs,
            patience=999,
            save_final_model=False,
            param_selection_mode=False,
            num_workers=1,  # why-the-ff should one need 6 workers for dataloading?!
            monitor_train=True,
            monitor_test=True,
        )
        return tagger


class BiLSTMConll03enPooled(FlairScoreTask):
    @staticmethod
    def build_train_sequence_tagger(corpus, tag_dictionary, params, TAG_TYPE="ner"):
        return build_and_train_conll03en_flair_sequence_tagger(
            corpus, TAG_TYPE, tag_dictionary
        )


class BiLSTMConll03en(FlairScoreTask):
    @staticmethod
    def build_train_sequence_tagger(
        corpus, tag_dictionary, params: Params, TAG_TYPE="ner"
    ):
        embeddings: StackedEmbeddings = StackedEmbeddings(
            embeddings=[
                WordEmbeddings("glove"),
                FlairEmbeddings("news-forward"),
                FlairEmbeddings("news-backward"),
            ]
        )
        from flair.models import SequenceTagger

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=TAG_TYPE,
        )

        from flair.trainers import ModelTrainer

        corpus = Corpus(train=corpus.train, dev=corpus.dev, test=[])
        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(
            "flair_checkpoints",
            train_with_dev=False,
            max_epochs=params.max_epochs,
            save_final_model=False,
        )  # original

        return tagger


if __name__ == "__main__":
    import os

    raw_data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )

    num_folds = 1
    data_supplier, splits = build_data_supplier_splits_trainset_only(
        raw_data_supplier, num_folds, 0.1
    )

    start = time()
    task = FlairGoveSeqTagScorer(
        params=Params(max_epochs=2), data_supplier=data_supplier
    )
    num_workers = 0  # min(multiprocessing.cpu_count() - 1, num_folds)
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=num_workers)
    print(
        " %d folds %d workers took: %0.2f seconds"
        % (num_folds, num_workers, time() - start)
    )
    pprint(m_scores_std_scores)
