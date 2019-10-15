from typing import List, Dict
from flair.data import Sentence, Token, Corpus, Dictionary
from torch.utils.data import Dataset
from util import data_io

TAG_TYPE = 'ner'
LABELS = ['Generic','Task','Method','Material','Metric','OtherScientificTerm']
def build_flair_sentences(d: Dict) -> List[Sentence]:

    def prefix_to_BIOES(label,start,end,current_index):
        if end - start > 0:
            if current_index == start:
                prefix = 'B'
            elif current_index == end:
                prefix = 'E'
            else:
                prefix = 'I'
        else:
            prefix = 'S'

        return prefix+'-'+label

    def tag_it(token:Token,index,ner_spans):
        labels = [(start,end,label) for start,end,label in ner_spans if index>=start and index<=end]

        if len(labels)>0:
            for start,end,label in labels:
                token.add_tag(TAG_TYPE, prefix_to_BIOES(label, start, end, index))
        else:
            token.add_tag(TAG_TYPE, 'O')

    offset = 0
    sentences = []
    for tokens, ner_spans in zip(d['sentences'], d['ner']):
        assert all([l in LABELS for _,_,l in ner_spans])
        sentence: Sentence = Sentence()
        [sentence.add_token(Token(tok)) for tok in tokens]
        [tag_it(token, k + offset, ner_spans) for k, token in enumerate(sentence)]
        offset += len(tokens)
        sentences.append(sentence)

    return sentences


def read_scierc_file_to_FlairSentences(
    jsonl_file:str
    )->Dataset:
    dataset:Dataset = [sent
                       for d in data_io.read_jsonl(jsonl_file)
                       for sent in build_flair_sentences(d)]
    return dataset

def get_scierc_data_as_flair_sentences(data_path):

    sentences = [sent
                 for jsonl_file in ['train.json','dev.json','test.json']
                 for sent in read_scierc_file_to_FlairSentences('%s/%s' % (data_path, jsonl_file))]
    return sentences

def build_tag_dict(sentences:List[Sentence],tag_type):
    corpus = Corpus(
        train=sentences,
        dev=[],
        test=[], name='scierc')

    # Make the tag dictionary
    tag_dictionary: Dictionary = Dictionary()
    # tag_dictionary.add_item("O")
    for sentence in corpus.get_all_sentences():
        for token in sentence.tokens:
            tag_dictionary.add_item(token.get_tag(tag_type).value)
    # tag_dictionary.add_item("<START>")
    # tag_dictionary.add_item("<STOP>")
    return corpus.make_tag_dictionary(tag_type)

if __name__ == '__main__':
    file = '../data/processed_data/json/dev.json'
    print(read_scierc_file_to_FlairSentences(file))

