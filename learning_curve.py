import multiprocessing
from itertools import groupby
from time import time
from typing import Dict, List, Tuple, Any, Iterable
from sklearn.model_selection import ShuffleSplit

import numpy as np
from benchmark_flair_tagger import score_flair_tagger
from util import data_io

from benchmark_spacyCrf_tagger import score_spacycrfsuite_tagger
from crossvalidation import calc_scores, calc_mean_and_std
from flair_scierc_ner import build_flair_sentences, build_tag_dict, TAG_TYPE


def get_flair_sentences(file):
    return [seq for d in data_io.read_jsonl(file) for seq in build_flair_sentences(d)]

len_train_set=None

def data_supplier():
    datasets = load_datasets()
    data = datasets['train'] + datasets['test']
    return data


def groupbykey(x:List[Dict]):
    return {k: [l for _, l in group]
            for k, group in
            groupby(sorted([(k, v) for d in x for k, v in d.items()], key=lambda x: x[0]), key=lambda x: x[0])
            }

def groupbyfirst(x:Iterable[Tuple[Any,Any]]):
    return {k: [l for _, l in group]
            for k, group in
            groupby(sorted([(k, v) for k,v in x], key=lambda x: x[0]), key=lambda x: x[0])
            }

from pathlib import Path
home = str(Path.home())
data_path = home+'/data/scierc_data/processed_data/json'
results_path = home+'/data/scierc_data'

def load_datasets():
    datasets = {dataset_name: get_flair_sentences('%s/%s.json' % (data_path, dataset_name)) for dataset_name in
                ['train', 'dev', 'test']}
    return datasets

def tuple_2_dict(t):
    m,s = t
    return {'mean':m,'std':s}


def calc_write_learning_curve(name,data_params_supplier,scorer_fun,splits,n_jobs=0):
    start = time()
    scores = calc_scores(data_params_supplier, scorer_fun, [split for train_size, split in splits], n_jobs=n_jobs)
    print('calculating learning-curve for %s took %0.2f seconds'%(name,time()-start))
    results = groupbyfirst(zip([train_size for train_size, _ in splits], scores))
    data_io.write_json(results_path + '/learning_curve_%s.json'%name, results)
    trainsize_to_mean_std_scores = {train_size: tuple_2_dict(calc_mean_and_std(m)) for train_size, m in results.items()}
    data_io.write_json(results_path + '/learning_curve_meanstd_%s.json'%name, trainsize_to_mean_std_scores)


if __name__ == '__main__':
    data = load_datasets()
    len_train_and_test = len(data['train']+data['test'])
    len_train_set = len(data['train'])

    num_folds = 3
    splits=[(train_size,{'train': train,'dev': train[:20] , 'test':list(range(len_train_set,len_train_and_test))})
     for train_size in np.arange(0.1,1.0,0.1).tolist()+[0.99]
     for train,_ in ShuffleSplit(n_splits=num_folds, train_size=train_size,test_size=None, random_state=111).split(
                X=range(len_train_set))
     ]
    print('got %d evaluations to calculate'%len(splits))

    data_params_supplier = lambda: {'data': data_supplier(),
                                    'params': {'max_epochs': 6},
                                    'tag_dictionary':build_tag_dict(data['train']+data['test'],TAG_TYPE)
                                    }
    calc_write_learning_curve('spacyCrfSuite',data_supplier,score_spacycrfsuite_tagger,splits,min(multiprocessing.cpu_count() - 1, len(splits)))
    calc_write_learning_curve('flair',data_params_supplier,score_flair_tagger,splits,0)
