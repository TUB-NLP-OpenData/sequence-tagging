import random
from pprint import pprint
import numpy as np

from util.util_methods import get_dict_paths, set_val, get_val
from util.worker_pool import WorkerPool

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
        data_supplier,
        score_fun,
        splits,
        n_jobs=0
    ):
    scores = calc_scores(data_supplier, score_fun, splits, n_jobs)
    assert len(scores) == len(splits)

    m_scores, std_scores = calc_mean_and_std(scores)
    return {'m_scores':m_scores,'std_scores':std_scores}


def task_fun_builder(score_fun, data_supplier):
    data = data_supplier()

    def fun(datum):
        return score_fun(datum, data)

    return fun

def calc_scores(data_supplier, score_fun, splits, n_jobs):
    if n_jobs > 0:
        with WorkerPool(processes=n_jobs,task_fun_builder=task_fun_builder,task_fun_kwargs={'data_supplier':data_supplier,'score_fun':score_fun}) as p:
            scores = [r for r in p.process_unordered(splits)]
    else:
        data = data_supplier()
        scores = [score_fun(split, data) for split in splits]
    return scores

if __name__ == '__main__':

    def dummy_score_fun(split,model_data):
        return {model_data:{'dummy-score-%s'%dataset_name:random.random() for dataset_name in split}}


    scores = calc_scores(data_supplier=lambda: 'some-model', score_fun=dummy_score_fun,
                         splits=[('train_%k', 'test_%k') for k in range(3)], n_jobs=2)
    pprint(scores)

    mscores = calc_mean_std_scores(data_supplier=lambda: 'some-model', score_fun=dummy_score_fun,
                         splits=[('train_%k', 'test_%k') for k in range(3)], n_jobs=2)
    pprint(mscores)