import os
from allennlp.data.dataset_readers import Conll2003DatasetReader
from reading_scierc_data import read_scierc_seqs


def read_scierc_data(data_path):
    return {dataset_name: read_scierc_seqs('%s/%s.json' % (data_path,dataset_name)) for dataset_name in ['train','dev','test']}

def read_JNLPBA_data(path):
    conll_reader = Conll2003DatasetReader()
    def read_file(file):
        instances = conll_reader.read(file)
        return [[(tok.text,tag) for tok,tag in zip(instance.fields['tokens'],instance.fields['tags'])] for instance in instances]
    return {file.split('.')[0]:list(read_file('%s/%s'%(path,file))) for file in os.listdir(path)}

if __name__ == '__main__':
    path = '../scibert/data/ner/JNLPBA'
    data  = read_JNLPBA_data(path)