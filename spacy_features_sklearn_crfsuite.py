import re
from collections import Counter
from pprint import pprint
from time import time
from typing import List, Tuple

import flair.datasets
import sklearn_crfsuite
import spacy
from spacy.tokenizer import Tokenizer

from reading_seqtag_data import read_JNLPBA_data
from seq_tag_util import bilou2bio, spanwise_pr_re_f1, calc_seqtag_tokenlevel_scores


class SpacyCrfSuiteTagger(object):
    def __init__(
        self,
        nlp=spacy.load("en_core_web_sm", disable=["parser"]),
        verbose=False,
        c1=None,
        c2=None,
    ):
        self.c1, self.c2 = c1, c2
        self.spacy_nlp = nlp
        infix_re = re.compile(r"\s")
        self.spacy_nlp.tokenizer = Tokenizer(
            nlp.vocab, infix_finditer=infix_re.finditer
        )
        self.verbose = verbose

    def fit(self, data: List[List[Tuple[str, str]]]):

        tag_counter = Counter([tag for sent in data for _, tag in sent])
        self.tag2count = {t: c for t, c in tag_counter.items() if t != "O"}
        # # print(tag2count)
        #
        # dictionary = Dictionary()
        # [dictionary.add_item(t) for t in tag2count]
        # dictionary.add_item('O')

        start = time()
        processed_data = [
            self.extract_features_with_spacy([token for token, tag in datum])
            for datum in data
        ]
        if self.verbose:
            print("spacy-processing train-data took: %0.2f" % (time() - start))

        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=200,
            all_possible_transitions=True,
        )
        targets = [[tag for token, tag in datum] for datum in data]
        start = time()
        self.crf.fit(processed_data, targets)
        if self.verbose:
            print("crfsuite-fitting took: %0.2f" % (time() - start))

    def extract_features_with_spacy(self, tokens: List[str]):
        text = " ".join(tokens)

        try:
            doc = self.spacy_nlp(text)
            assert len(doc) == len(tokens)
            features = [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    # 'dep':token.dep_,
                    "shape": token.shape_,
                    "is_alpha": token.is_alpha,
                    "is_stop": token.is_stop,
                }
                for token in doc
            ]
        except BaseException:
            features = [{"text": ""}]
        return features

    def predict(self, data):
        processed_data = [self.extract_features_with_spacy(datum) for datum in data]
        y_pred = self.crf.predict(processed_data)
        return y_pred

    def predict_marginals(self, data):
        processed_data = [self.extract_features_with_spacy(datum) for datum in data]
        probas = self.crf.predict_marginals(processed_data)
        return probas


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

    from pathlib import Path

    home = str(Path.home())
    # data_path = home+'/data/scierc_data/processed_data/json/'
    # datasets = read_scierc_data(data_path)

    path = "../scibert/data/ner/JNLPBA"
    datasets = read_JNLPBA_data(path)

    train_data, test_data = datasets["train"], datasets["test"]
    print("train/test-set-len: %d / %d" % (len(train_data), len(test_data)))

    tagger = SpacyCrfSuiteTagger(c1=0.5, c2=0.0)
    tagger.fit(train_data)

    y_pred = tagger.predict([[token for token, tag in datum] for datum in train_data])
    y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
    targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]
    pprint(Counter([t for tags in targets for t in tags]))
    pprint(
        "train-f1-macro: %0.2f"
        % calc_seqtag_tokenlevel_scores(targets, y_pred)["f1-macro"]
    )
    pprint(
        "train-f1-micro: %0.2f"
        % calc_seqtag_tokenlevel_scores(targets, y_pred)["f1-micro"]
    )
    _, _, f1 = spanwise_pr_re_f1(y_pred, targets)
    pprint("train-f1-spanwise: %0.2f" % f1)

    y_pred = tagger.predict([[token for token, tag in datum] for datum in test_data])
    y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
    targets = [bilou2bio([tag for token, tag in datum]) for datum in test_data]
    pprint(
        "test-f1-macro: %0.2f"
        % calc_seqtag_tokenlevel_scores(targets, y_pred)["f1-macro"]
    )
    pprint(
        "test-f1-micro: %0.2f"
        % calc_seqtag_tokenlevel_scores(targets, y_pred)["f1-micro"]
    )
    _, _, f1 = spanwise_pr_re_f1(y_pred, targets)
    pprint("test-f1-spanwise: %0.2f" % f1)

"""
# UD_English_data
spacy-processing train-data took: 66.69
crfsuite-fitting took: 31.05
    'test-f1-macro: 0.70'
# SCIERC    
'train-f1-macro: 0.76'
'train-f1-micro: 0.91'
'train-f1-spanwise: 0.73'
'test-f1-macro: 0.53'
'test-f1-micro: 0.82'
'test-f1-spanwise: 0.48'

# scierc
train/test-set-len: 1861 / 551
'train-f1-macro: 0.90'
'train-f1-micro: 0.96'
'train-f1-spanwise: 0.86'
'test-f1-macro: 0.54'
'test-f1-micro: 0.82'
'test-f1-spanwise: 0.49'

# JNLPBA
train/test-set-len: 16807 / 3856
'train-f1-macro: 0.86'
'train-f1-micro: 0.95'
'train-f1-spanwise: 0.81'
'test-f1-macro: 0.69'
'test-f1-micro: 0.91'
'test-f1-spanwise: 0.63'
"""
