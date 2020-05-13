import os
from flair.data import Corpus

from flair.datasets import ColumnCorpus
from flair.embeddings import (
    WordEmbeddings,
    StackedEmbeddings,
    PooledFlairEmbeddings,
)

"""
    based on: "flair/resources/docs/EXPERIMENTS.md"
"""


def build_conll03en_corpus(base_path: str):
    document_as_sequence = False
    corpus = ColumnCorpus(
        base_path,
        column_format={0: "text", 1: "pos", 2: "np", 3: "ner"},
        train_file="train.txt",
        dev_file="dev.txt",
        test_file="test.txt",
        tag_to_bioes="ner",
        document_separator_token=None if not document_as_sequence else "-DOCSTART-",
    )
    tag_type = "ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    return corpus, tag_type, tag_dictionary


def build_and_train_conll03en_flair_sequence_tagger(corpus,tag_type,tag_dictionary):
    '''
    do not change!
    same configuration as described in
      file:  "flair/resources/docs/EXPERIMENTS.md"
      section: "CoNLL-03 Named Entity Recognition (English)"
    '''
    embeddings: StackedEmbeddings = StackedEmbeddings(
        embeddings=[
            WordEmbeddings("glove"),
            PooledFlairEmbeddings("news-forward", pooling="min"),
            PooledFlairEmbeddings("news-backward", pooling="min"),
        ]
    )
    from flair.models import SequenceTagger
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
    )

    from flair.trainers import ModelTrainer

    corpus = Corpus(train=corpus.train, dev=corpus.dev,test=[])
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # trainer.train("resources/taggers/example-ner", train_with_dev=True, max_epochs=150) # original
    trainer.train("resources/taggers/example-ner", train_with_dev=False, max_epochs=40,save_final_model=False) # original

    return tagger


if __name__ == "__main__":
    HOME = os.environ["HOME"] #+ "/hpc"
    base_path = HOME + "/FARM/data/conll03-en"

    corpus, tag_type, tag_dictionary = build_conll03en_corpus(base_path)

    tagger = build_and_train_conll03en_flair_sequence_tagger(corpus,tag_type,tag_dictionary)

