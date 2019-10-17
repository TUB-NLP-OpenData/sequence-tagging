import logging
import multiprocessing
import os
import shutil
from pprint import pprint
from time import time
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from sklearn.model_selection import ShuffleSplit

from crossvalidation import calc_mean_std_scores, ScoreTask
from reading_scierc_data import TAG_TYPE, build_tag_dict, read_scierc_seqs, build_flair_sentences_from_sequences
from seq_tag_util import bilou2bio, calc_seqtag_f1_scores

def score_flair_tagger(
        split,
        data,tag_dictionary,params,train_dev_test_sentences_builder

):
    from flair.trainers import ModelTrainer, trainer
    logger = trainer.log
    logger.setLevel(logging.WARNING)
    # torch.cuda.empty_cache()

    train_sentences,dev_sentences,test_sentences = train_dev_test_sentences_builder(split, data)

    corpus = Corpus(
        train=train_sentences,
        dev=dev_sentences,
        test=test_sentences, name='scierc')

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=64,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=TAG_TYPE,
                                            locked_dropout=0.01,
                                            dropout=0.01,
                                            use_crf=True)
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.RMSprop)
    # print(tagger)
    # pprint([p_name for p_name, p in tagger.named_parameters()])
    save_path = 'flair_seq_tag_model_%s'%str(multiprocessing.current_process())
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    assert not os.path.isdir(save_path)
    trainer.train(base_path=save_path,
                  learning_rate=0.01,
                  mini_batch_size=32,
                  max_epochs=params['max_epochs'],
                  patience=999,
                  save_final_model=False,
                  param_selection_mode=True,
                  num_workers=1# why-the-ff should one need 6 workers for dataloading?!
                  )
    # plotter = Plotter()
    # plotter.plot_training_curves('%s/loss.tsv' % save_path)
    # plotter.plot_weights('%s/weights.txt' % save_path)

    def flair_tagger_predict_bio(sentences: List[Sentence]):
        train_data = [[(token.text, token.tags[tagger.tag_type].value) for token in datum] for datum in sentences]
        targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]

        pred_sentences = tagger.predict(sentences)
        pred_data = [bilou2bio([token.tags[tagger.tag_type].value for token in datum]) for datum in pred_sentences]
        return pred_data,targets

    return {
        'train':calc_seqtag_f1_scores(flair_tagger_predict_bio,train_sentences),
        'test':calc_seqtag_f1_scores(flair_tagger_predict_bio,test_sentences)
    }

def train_dev_test_sentences_builder(split, data):
    return [build_flair_sentences_from_sequences([data[i] for i in split[dataset_name]]) for dataset_name in ['train', 'dev', 'test']]

def kwargs_builder(data_path):
    sentences = [sent
            for jsonl_file in ['train.json','dev.json','test.json']
            for sent in read_scierc_seqs('%s/%s' % (data_path, jsonl_file))]

    return {'data': sentences,
     'params': {'max_epochs': 1},
     'tag_dictionary': build_tag_dict(sentences, TAG_TYPE),
     'train_dev_test_sentences_builder': train_dev_test_sentences_builder}

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    data_path = home + '/data/scierc_data/processed_data/json/'
    sentences = [sent
            for jsonl_file in ['train.json','dev.json','test.json']
            for sent in read_scierc_seqs('%s/%s' % (data_path, jsonl_file))]

    num_folds = 2
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=111)
    splits = [{'train':train[:30],'dev':train[:round(len(train)/5)],'test':test} for train,test in splitter.split(X=range(len(sentences)))]


    start = time()
    n_jobs = 0#min(5, num_folds)

    task = ScoreTask(score_flair_tagger,kwargs_builder,{'data_path':data_path})
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=n_jobs)
    print('flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds'%(num_folds,n_jobs,time()-start))
    pprint(m_scores_std_scores)
