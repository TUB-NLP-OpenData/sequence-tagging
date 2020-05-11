from typing import Dict, Callable

import torch
from flair.embeddings import StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from flair_conll2003_en import build_and_train_conll03en_flair_sequence_tagger
from flair_util import FlairScoreTask


class FlairGoveSeqTagScorer(FlairScoreTask):

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

class BiLSTMConll03en(FlairScoreTask):
    @staticmethod
    def build_train_sequence_tagger(corpus, tag_dictionary, params, TAG_TYPE="ner"):
        return build_and_train_conll03en_flair_sequence_tagger(corpus,TAG_TYPE,tag_dictionary)