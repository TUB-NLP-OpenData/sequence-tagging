import os
from functools import partial
from pprint import pprint
from time import time

from flair_score_tasks import FlairGoveSeqTagScorer
from eval_jobs import shufflesplit_trainset_only
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import read_JNLPBA_data

if __name__ == "__main__":
    from pathlib import Path

    home = str(Path.home())
    from json import encoder

    encoder.FLOAT_REPR = lambda o: format(o, ".2f")

    data_supplier = partial(
        read_JNLPBA_data, path=os.environ["HOME"] + "/scibert/data/ner/JNLPBA"
    )
    dataset = data_supplier()
    num_folds = 1

    splits = shufflesplit_trainset_only(dataset, num_folds)
    num_folds = len(splits)
    n_jobs = 0  # min(5, num_folds)# needs to be zero if using Transformers

    exp_name = 'flair-glove'
    task = FlairGoveSeqTagScorer(params={"max_epochs": 1}, data_supplier=data_supplier)
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
