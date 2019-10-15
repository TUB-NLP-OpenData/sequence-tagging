import multiprocessing
import sys

from util import data_io

sys.path.append('.')

from time import time

from flair_scierc_ner import build_flair_sentences
from seq_tag_util import bilou2bio, calc_seqtag_f1_scores
from spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger
from sklearn.model_selection import ShuffleSplit

from crossvalidation import calc_mean_std_scores

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from pprint import pprint


def score_spacycrfsuite_tagger(splits,params,datasets_builder_fun):
    data_splits = datasets_builder_fun(splits)

    def get_data_of_split(split_name):
        return [[(token.text, token.tags['ner'].value) for token in datum] for datum in data_splits[split_name]]

    tagger = SpacyCrfSuiteTagger(**params)
    tagger.fit(get_data_of_split('train'))

    def pred_fun(token_tag_sequences):
        y_pred = tagger.predict([[token for token, tag in datum] for datum in token_tag_sequences])
        y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
        targets = [bilou2bio([tag for token, tag in datum]) for datum in token_tag_sequences]
        return y_pred,targets

    return {split_name: calc_seqtag_f1_scores(pred_fun,get_data_of_split(split_name)) for split_name in data_splits.keys()}


from pathlib import Path

def get_scierc_data_as_flair_sentences():
    home = str(Path.home())
    data_path = home + '/data/scierc_data/processed_data/json/'

    sentences = [sent for jsonl_file in ['train.json','dev.json','test.json']
                 for d in data_io.read_jsonl('%s/%s' % (data_path,jsonl_file))
                 for sent in build_flair_sentences(d)]
    return sentences


def spacyCrfSuite_kwargs_supplier():
    data = get_scierc_data_as_flair_sentences()
    return {
        'params':{'c1':0.5,'c2':0.0},
        'datasets_builder_fun': lambda split: {dataset_name:[data[i] for i in indizes] for dataset_name,indizes in split.items()}
    }

if __name__ == '__main__':


    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    sentences = get_scierc_data_as_flair_sentences()
    num_folds = 3
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=111)
    splits = [{'train':train,'dev':train[:round(len(train)/5)],'test':test} for train,test in splitter.split(X=range(len(sentences)))]

    start = time()
    m_scores_std_scores = calc_mean_std_scores(spacyCrfSuite_kwargs_supplier, score_spacycrfsuite_tagger, splits, n_jobs=min(multiprocessing.cpu_count() - 1, num_folds))
    print('spacy+crfsuite-tagger %d folds-PARALLEL took: %0.2f seconds'%(num_folds,time()-start))
    pprint(m_scores_std_scores)


'''
#############################################################################
on x1-carbon

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
'''