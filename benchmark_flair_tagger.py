import logging

import os
from functools import partial
from pprint import pprint
from time import time

from flair_score_tasks import FlairGoveSeqTagScorer, BiLSTMConll03en
from eval_jobs import shufflesplit_trainset_only, preserve_train_dev_test
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import read_JNLPBA_data, read_conll03_en

logging.getLogger("flair").setLevel(logging.INFO)

if __name__ == "__main__":
    from pathlib import Path

    home = str(Path.home())
    from json import encoder

    encoder.FLOAT_REPR = lambda o: format(o, ".2f")

    # data_supplier = partial(
    #     read_JNLPBA_data, path=os.environ["HOME"] + "/hpc/scibert/data/ner/JNLPBA"
    # )

    data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )
    dataset = data_supplier()
    num_folds = 1

    splits = preserve_train_dev_test(dataset, num_folds)
    n_jobs = 0  # min(5, num_folds)# needs to be zero if using Transformers

    exp_name = 'flair'
    task = BiLSTMConll03en(params={}, data_supplier=data_supplier)
    start = time()
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=n_jobs)
    duration = time() - start
    print(
        "flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds"
        % (num_folds, n_jobs, duration)
    )
    exp_results = {
        "scores": m_scores_std_scores,
        "overall-time": duration,
        "num-folds": num_folds,
    }
    pprint(exp_results)
    # data_io.write_json("%s.json" % exp_name, exp_results)

    """
    ### conll03-en ###
    flair-tagger 1 folds with 0 jobs in PARALLEL took: 396.39 seconds
    {'num-folds': 1,
     'overall-time': 396.3929715156555,
     'scores': {'m_scores': {'dev': {'f1-micro-spanlevel': 0.8217027215631543,
                                     'seqeval-f1': 0.8136667237540675},
                             'test': {'f1-micro-spanlevel': 0.7936594698004921,
                                      'seqeval-f1': 0.7839528234453181},
                             'train': {'f1-micro-spanlevel': 0.8334028494281388,
                                       'seqeval-f1': 0.8261384943641026}},
                'std_scores': {'dev': {'f1-micro-spanlevel': 0.0,
                                       'seqeval-f1': 0.0},
                               'test': {'f1-micro-spanlevel': 0.0,
                                        'seqeval-f1': 0.0},
                               'train': {'f1-micro-spanlevel': 0.0,
                                         'seqeval-f1': 0.0}}}}


 
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
