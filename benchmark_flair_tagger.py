import os
from functools import partial
from time import time
from typing import Dict, Callable

import torch
from flair.embeddings import (
    StackedEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from eval_jobs import crosseval_on_concat_dataset
from flair_util import FlairScoreTask
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import read_JNLPBA_data


class FlairGoveSeqTagScorer(FlairScoreTask):
    def __init__(self, params: Dict, data_supplier: Callable) -> None:
        super().__init__(params, data_supplier)

    @staticmethod
    def build_train_sequence_tagger(corpus, tag_dictionary, params, TAG_TYPE="ner"):
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
            max_epochs=params["max_epochs"],
            patience=999,
            save_final_model=False,
            param_selection_mode=False,
            num_workers=1,  # why-the-ff should one need 6 workers for dataloading?!
            monitor_train=True,
            monitor_test=True,
        )
        return tagger


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

    splits = crosseval_on_concat_dataset(dataset, num_folds)
    start = time()
    num_folds = len(splits)
    n_jobs = 0  # min(5, num_folds)# needs to be zero if using Transformers

    task = FlairGoveSeqTagScorer(params={"max_epochs": 2}, data_supplier=data_supplier)
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=n_jobs)
    duration = time() - start
    print(
        "flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds"
        % (num_folds, n_jobs, duration)
    )
    # exp_results = {
    #     "scores": m_scores_std_scores,
    #     "overall-time": duration,
    #     "num-folds": num_folds,
    # }
    # data_io.write_json("%s.json" % exp_name, exp_results)

    """
    flair-tagger 3 folds with 3 jobs in PARALLEL took: 4466.98 seconds
    {'m_scores': {'test': {'f1-micro-spanlevel': 0.6571898523684608,

    crosseval 20 epochs
    flair-tagger 3 folds with 3 jobs in PARALLEL took: 4383.46 seconds
    {'m_scores': {'test': {'f1-micro-spanlevel': 0.663033472415799,

    crosseval with Bert, 2 epochs
      "overall-time": 2111.6508309841156,
      "num-folds": 3
    "f1-micro-spanlevel": 0.6909809545678615
    """
