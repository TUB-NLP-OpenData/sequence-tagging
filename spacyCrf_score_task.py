import os
from functools import partial
from pprint import pprint
from time import time
from typing import Dict, Any, Tuple, List

from torch import multiprocessing

from eval_jobs import (
    crosseval_on_concat_dataset,
    TaggedSeqsDataSet,
    shufflesplit_trainset_only,
)
from experiment_util import split_data, split_splits, SeqTagScoreTask, SeqTagTaskData
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import read_JNLPBA_data, read_conll03_en
from seq_tag_util import bilou2bio, Sequences
from spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger, Params
from util import data_io


def spacycrf_predict_bio(tagger, token_tag_sequences) -> Tuple:
    y_pred = tagger.predict(
        [[token for token, tag in datum] for datum in token_tag_sequences]
    )
    y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
    targets = [
        bilou2bio([tag for token, tag in datum]) for datum in token_tag_sequences
    ]
    return y_pred, targets


class SpacyCrfScorer(SeqTagScoreTask):
    @staticmethod
    def build_task_data(**task_params) -> SeqTagTaskData:
        return SeqTagTaskData(**build_task_data_maintaining_splits(**task_params))

    @classmethod
    def predict_with_targets(
        cls, splits: Dict[str, List], params: Params
    ) -> Dict[str, Tuple[Sequences, Sequences]]:

        tagger = SpacyCrfSuiteTagger(params=params)
        tagger.fit(splits["train"])

        return {
            split_name: spacycrf_predict_bio(tagger, split_data)
            for split_name, split_data in splits.items()
        }


from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, ".2f")


def build_kwargs(data_supplier, params):
    dataset: TaggedSeqsDataSet = data_supplier()
    data = dataset.train + dataset.dev + dataset.test

    return {
        "params": params,
        "data": data,
        "split_fun": split_data,
    }


def build_task_data_maintaining_splits(params, data_supplier):
    data: TaggedSeqsDataSet = data_supplier()
    return {
        "data": data._asdict(),
        "params": params,
        "split_fun": split_splits,
    }


if __name__ == "__main__":
    import os

    data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )
    dataset = data_supplier()
    num_folds = 1

    splits = shufflesplit_trainset_only(dataset, num_folds, train_size=0.1)

    start = time()
    task = SpacyCrfScorer(params=Params(c1=0.5, c2=0.0), data_supplier=data_supplier)
    num_workers = 0  # min(multiprocessing.cpu_count() - 1, num_folds)
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
