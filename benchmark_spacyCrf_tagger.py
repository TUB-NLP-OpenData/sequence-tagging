import os
import sys
from functools import partial
from pprint import pprint

from torch import multiprocessing

from eval_jobs import crosseval_on_concat_dataset, TaggedSeqsDataSet
from reading_seqtag_data import read_JNLPBA_data
from experiment_util import split_data, split_splits
from util import data_io

from time import time

from seq_tag_util import bilou2bio, calc_seqtag_f1_scores
from spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger
from sklearn.model_selection import ShuffleSplit

from mlutil.crossvalidation import calc_mean_std_scores, ScoreTask


def score_spacycrfsuite_tagger(splits, params, datasets_builder_fun, data):
    data_splits = datasets_builder_fun(splits, data)

    tagger = SpacyCrfSuiteTagger(**params)
    tagger.fit(data_splits["train"])

    def pred_fun(token_tag_sequences):
        y_pred = tagger.predict(
            [[token for token, tag in datum] for datum in token_tag_sequences]
        )
        y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
        targets = [
            bilou2bio([tag for token, tag in datum]) for datum in token_tag_sequences
        ]
        return y_pred, targets

    return {
        split_name: calc_seqtag_f1_scores(pred_fun, data_splits[split_name])
        for split_name in data_splits.keys()
    }


from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, ".2f")


def build_kwargs(data_supplier, params):
    dataset: TaggedSeqsDataSet = data_supplier()
    data = dataset.train + dataset.dev + dataset.test

    return {
        "params": params,
        "data": data,
        "datasets_builder_fun": split_data,
    }

def kwargs_builder_maintaining_train_dev_test(params, data_supplier):
    data: TaggedSeqsDataSet = data_supplier()
    return {
        "data": data._asdict(),
        "params": params,
        "datasets_builder_fun": split_splits,
    }


if __name__ == "__main__":

    data_supplier = partial(
        read_JNLPBA_data, path=os.environ["HOME"] + "/scibert/data/ner/JNLPBA"
    )
    dataset = data_supplier()
    num_folds = 3

    splits = crosseval_on_concat_dataset(dataset, num_folds, test_size=0.2)

    start = time()
    task = ScoreTask(
        score_fun=score_spacycrfsuite_tagger,
        build_kwargs_fun=build_kwargs,
        builder_kwargs={
            "params": {"c1": 0.5, "c2": 0.0},
            "data_supplier": data_supplier,
        },
    )
    num_workers = min(multiprocessing.cpu_count() - 1, num_folds)
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=num_workers)
    print(
        "spacy+crfsuite-tagger %d folds %d workers took: %0.2f seconds"
        % (num_folds, num_workers, time() - start)
    )
    pprint(m_scores_std_scores)
    data_io.write_json("spacy-crf-scores.json", m_scores_std_scores)

"""
#############################################################################
on x1-carbon scierc-data

spacy+crfsuite-tagger 3 folds-PARALLEL took: 74.86 seconds
{'m_scores': {'dev': {'f1-macro': 0.8822625032484681,
                      'f1-micro': 0.9528343173272004,
                      'f1-spanwise': 0.8470436086284675},
              'test': {'f1-macro': 0.5742946309433821,
                       'f1-micro': 0.832899550463387,
                       'f1-spanwise': 0.5345123493111902},
              'train': {'f1-macro': 0.8844589822247658,
                        'f1-micro': 0.9522832740014087,
                        'f1-spanwise': 0.842115934181045}},
 'std_scores': {'dev': {'f1-macro': 0.009338633769168965,
                        'f1-micro': 0.0020278574245488883,
                        'f1-spanwise': 0.007549419792021609},
                'test': {'f1-macro': 0.019814579353249383,
                         'f1-micro': 0.003716883368130915,
                         'f1-spanwise': 0.00622111159374358},
                'train': {'f1-macro': 0.002357299459853917,
                          'f1-micro': 0.0006117207995410837,
                          'f1-spanwise': 0.0025053074858217297}}}
                          
# on gunther and JNLPBA-data
spacy+crfsuite-tagger 3 folds-PARALLEL took: 507.72 seconds

{'m_scores': {'dev': {'f1-macro': 0.8645722098040851,
                      'f1-micro': 0.950338415398806,
                      'f1-spanwise': 0.8067537304384289},
              'test': {'f1-macro': 0.759263780405201,
                       'f1-micro': 0.9214014646164155,
                       'f1-spanwise': 0.7142331137117344},
              'train': {'f1-macro': 0.8634555655809333,
                        'f1-micro': 0.9501748452465374,
                        'f1-spanwise': 0.8063675130764999}},
 'std_scores': {'dev': {'f1-macro': 0.004099180910042565,
                        'f1-micro': 0.0003087015574075688,
                        'f1-spanwise': 0.0024466403296221633},
                'test': {'f1-macro': 0.00042005226966156795,
                         'f1-micro': 0.0004021343926907527,
                         'f1-spanwise': 0.0019832653200119116},
                'train': {'f1-macro': 0.0004980035249218241,
                          'f1-micro': 0.00024561871064496466,
                          'f1-spanwise': 0.0007059342785105874}}}

"""
