import logging
import multiprocessing
from pprint import pprint
from time import time
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from sklearn.model_selection import ShuffleSplit

from crossvalidation import calc_mean_std_scores
from flair_scierc_ner import TAG_TYPE, get_scierc_data_as_flair_sentences, build_tag_dict
from seq_tag_util import bilou2bio, calc_seqtag_f1_scores

def score_flair_tagger(
        splits,
        data_params,

):
    data = data_params['data']
    params = data_params['params']

    from flair.trainers import ModelTrainer, trainer
    logger = trainer.log
    logger.setLevel(logging.WARNING)

    data_splits = {split_name:[data[split_name][i] for i in split] for split_name,split in splits.items()}

    train_sentences,dev_sentences,test_sentences = data_splits['train'],data_splits['dev'],data_splits['test'],

    corpus = Corpus(
        train=train_sentences,
        dev=dev_sentences,
        test=test_sentences, name='scierc')
    tag_dictionary = data_params['tag_dictionary']

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
    save_path = 'flair_sequence_tagging/scierc-ner-%s'%multiprocessing.current_process()
    trainer.train(base_path='%s' % save_path,
                  # evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.01,
                  mini_batch_size=32,
                  max_epochs=params['max_epochs'],
                  patience=3,
                  save_final_model=False
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
        'train':calc_seqtag_f1_scores(flair_tagger_predict_bio,corpus.train),
        'test':calc_seqtag_f1_scores(flair_tagger_predict_bio,corpus.test)
    }


if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    data_path = home + '/data/scierc_data/processed_data/json/'
    sentences = get_scierc_data_as_flair_sentences(data_path =data_path)
    num_folds = 1
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.9, random_state=111)
    splits = [{'train':train,'dev':train[:round(len(train)/5)],'test':test} for train,test in splitter.split(X=range(len(sentences)))]


    start = time()
    n_jobs = 0#min(5, num_folds)

    data_params_supplier = lambda: {'data': get_scierc_data_as_flair_sentences(data_path),
                                    'params':{'max_epochs': 1},
                                    'tag_dictionary':build_tag_dict(sentences,TAG_TYPE)
                                    }
    m_scores_std_scores = calc_mean_std_scores(data_params_supplier, score_flair_tagger, splits, n_jobs=n_jobs)
    print('flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds'%(num_folds,n_jobs,time()-start))
    pprint(m_scores_std_scores)
