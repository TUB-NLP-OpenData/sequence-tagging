import random
from pprint import pprint
from typing import Iterable, List, Any

import numpy as np

from util.util_methods import get_dict_paths, set_val, get_val
from util.worker_pool import WorkerPool, Task
import warnings
warnings.filterwarnings('ignore')

def imputed(x):
    return np.NaN if isinstance(x,str) else x

def calc_mean_and_std(eval_metrices):
    assert isinstance(eval_metrices, list) and all([isinstance(d, dict) for d in eval_metrices])
    paths = []
    get_dict_paths(paths, [], eval_metrices[0])
    means = {}
    stds = {}
    for p in paths:
        try:
            m_val = np.mean([imputed(get_val(d,p)) for d in eval_metrices])
            set_val(means,p,m_val)
        except:
            print(p)
        try:
            std_val = np.std([imputed(get_val(d,p)) for d in eval_metrices])
            set_val(stds,p,std_val)
        except:
            print(p)

    return means,stds

def calc_mean_std_scores(
        score_task:Task,
        scoring_jobs,
        n_jobs=0
    ):
    scores = calc_scores(score_task, scoring_jobs,n_jobs)
    assert len(scores) == len(scoring_jobs)

    m_scores, std_scores = calc_mean_and_std(scores)
    return {'m_scores':m_scores,'std_scores':std_scores}

def calc_scores(score_task:Task, scoring_jobs:List[Any], n_jobs):
    if n_jobs > 0:
        with WorkerPool(processes=n_jobs,
                        task=score_task,
                        daemons=False) as p:
            scores = [r for r in p.process_unordered(scoring_jobs)]
    else:
        with score_task as task:
            scores = [task(job) for job in scoring_jobs]
    assert len(scores)==len(scoring_jobs)
    assert all(s is not None for s in scores)
    return scores

def dummy_score_fun(split, model_data):
    return {model_data: {'dummy-score-%s' % dataset_name: random.random() for dataset_name in split}}

def kwargs_builder(param):
    return {'model_data': 'some-model-%s' % param}

class ScoreTask(Task):

    def __init__(self,score_fun,kwargs_builder,params) -> None:
        super().__init__()
        self.score_fun = score_fun
        self.kwargs_builder = kwargs_builder
        self.params = params

    def __enter__(self):
        self.kwargs = self.kwargs_builder(self.params)
        return self

    def __call__(self, data):
        return self.score_fun(data,**self.kwargs)


if __name__ == '__main__':

    task = ScoreTask(dummy_score_fun, kwargs_builder, 'testparam')
    scores = calc_scores(task,
                         scoring_jobs=[('train_%k', 'test_%k') for k in range(3)], n_jobs=2)
    pprint(scores)

    mscores = calc_mean_std_scores(task,
                         scoring_jobs=[('train_%k', 'test_%k') for k in range(3)], n_jobs=2)
    pprint(mscores)