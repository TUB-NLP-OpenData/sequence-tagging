import logging
import multiprocessing
import os
import shutil
from functools import partial
from pprint import pprint
from time import time
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    BertEmbeddings,
)
from flair.models import SequenceTagger
from sklearn.model_selection import ShuffleSplit

from mlutil.crossvalidation import calc_mean_std_scores, ScoreTask
from reading_scierc_data import (
    TAG_TYPE,
    build_tag_dict,
    build_flair_sentences_from_sequences,
)
from reading_seqtag_data import (
    get_JNLPBA_sequences,
    TaggedSeqsDataSet,
    read_JNLPBA_data,
    read_germEval_2014_data,
)
from seq_tag_util import bilou2bio, calc_seqtag_f1_scores


def score_flair_tagger(
    split, data, tag_dictionary, params, train_dev_test_sentences_builder
):
    from flair.trainers import ModelTrainer, trainer

    logger = trainer.log
    logger.setLevel(logging.WARNING)
    # torch.cuda.empty_cache()

    train_sentences, dev_sentences, test_sentences = train_dev_test_sentences_builder(
        split, data
    )

    corpus = Corpus(train=train_sentences, dev=dev_sentences, test=test_sentences)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("glove"),
        # BertEmbeddings("bert-base-multilingual-cased", layers="-1")
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
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
    process_name = multiprocessing.current_process().name
    save_path = "flair_seq_tag_model_%s" % process_name
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    assert not os.path.isdir(save_path)
    trainer.train(
        base_path=save_path,
        learning_rate=0.001,
        mini_batch_size=128,
        max_epochs=params["max_epochs"],
        patience=999,
        save_final_model=False,
        param_selection_mode=False,
        num_workers=1,  # why-the-ff should one need 6 workers for dataloading?!
        monitor_train=True,
        monitor_test=True,
    )

    def flair_tagger_predict_bio(sentences: List[Sentence]):
        train_data = [
            [(token.text, token.tags[tagger.tag_type].value) for token in datum]
            for datum in sentences
        ]
        targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]

        pred_sentences = tagger.predict(sentences)
        pred_data = [
            bilou2bio([token.tags[tagger.tag_type].value for token in datum])
            for datum in pred_sentences
        ]
        return pred_data, targets

    return {
        "train": calc_seqtag_f1_scores(flair_tagger_predict_bio, train_sentences),
        "test": calc_seqtag_f1_scores(flair_tagger_predict_bio, test_sentences),
    }


def kwargs_builder_maintaining_train_dev_test(params, data_supplier):
    data: TaggedSeqsDataSet = data_supplier()

    def train_dev_test_sentences_builder(split, data):
        return [
            build_flair_sentences_from_sequences(
                [getattr(data, dataset_name)[i] for i in split[dataset_name]]
            )
            for dataset_name in ["train", "dev", "test"]
        ]

    return {
        "data": data,
        "params": params,
        "tag_dictionary": build_tag_dict(
            [seq for seqs in data._asdict().values() for seq in seqs], TAG_TYPE
        ),
        "train_dev_test_sentences_builder": train_dev_test_sentences_builder,
    }


def kwargs_builder(params, data_supplier):
    dataset: TaggedSeqsDataSet = data_supplier()
    sentences = dataset.train + dataset.dev + dataset.test

    def train_dev_test_sentences_builder(split, data):
        return [
            build_flair_sentences_from_sequences([data[i] for i in split[dataset_name]])
            for dataset_name in ["train", "dev", "test"]
        ]

    return {
        "data": sentences,
        "params": params,
        "tag_dictionary": build_tag_dict(sentences, TAG_TYPE),
        "train_dev_test_sentences_builder": train_dev_test_sentences_builder,
    }


def shufflesplit_trainset_only(dataset, num_folds):
    splitter = ShuffleSplit(n_splits=num_folds, train_size=0.8, random_state=111)
    splits = [
        {
            "train": train,
            "dev": list(range(len(dataset.dev))),
            "test": list(range(len(dataset.test))),
        }
        for train, _ in splitter.split(X=range(len(dataset.train)))
    ]
    return kwargs_builder_maintaining_train_dev_test, splits


def preserve_train_dev_test(dataset, num_folds):
    splits = [
        {
            dsname: list(range(len(getattr(dataset, dsname))))
            for dsname in ["train", "dev", "test"]
        }
    ] * num_folds
    return kwargs_builder_maintaining_train_dev_test, splits


def crosseval_on_concat_dataset(dataset, num_folds):
    sentences = dataset.train + dataset.dev + dataset.test
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=111)
    splits = [
        {"train": train, "dev": train[: round(len(train) / 5)], "test": test}
        for train, test in splitter.split(X=range(len(sentences)))
    ]
    return kwargs_builder, splits


if __name__ == "__main__":
    from pathlib import Path

    home = str(Path.home())
    from json import encoder

    encoder.FLOAT_REPR = lambda o: format(o, ".2f")

    data_supplier = partial(
        read_JNLPBA_data, path=os.environ["HOME"] + "/scibert/data/ner/JNLPBA"
    )
    dataset = data_supplier()
    num_folds = 3
    # eval_mode = "test-preserving-crosseval"
    eval_mode = "crosseval"

    if eval_mode == "crosseval":
        kwargs_builder_fun, splits = crosseval_on_concat_dataset(dataset, num_folds)
    elif eval_mode == "test-preserving-crosseval":
        kwargs_builder_fun, splits = shufflesplit_trainset_only(dataset, num_folds)
    else:
        kwargs_builder_fun, splits = preserve_train_dev_test(dataset, num_folds)

    start = time()
    n_jobs = min(5, num_folds)

    kwargs_kwargs = {"params": {"max_epochs": 2}, "data_supplier": data_supplier}
    task = ScoreTask(score_flair_tagger, kwargs_builder_fun, kwargs_kwargs)
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=n_jobs)
    print(
        "flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds"
        % (num_folds, n_jobs, time() - start)
    )
    pprint(m_scores_std_scores)
