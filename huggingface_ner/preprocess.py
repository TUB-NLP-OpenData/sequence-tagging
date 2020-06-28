import argparse
import sys
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer

def read_and_preprocess(file:str):
    subword_len_counter = 0
    with open(file, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                yield line
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                yield ""
                yield line
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len
            yield line


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    # parser.add_argument(
    #     "--data_dir",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    # )
    args = parser.parse_args()
    return args

def get_label(s:str):
    x = s.split(' ')
    if len(x)==2:
        label = x[1]
    else:
        label = None
    return label

if __name__ == '__main__':
    args = build_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    max_len = args.max_seq_length
    max_len -= tokenizer.num_special_tokens_to_add()
    label_counter = Counter()

    def count_and_return(l:str):
        label = get_label(l)
        if label is not None:
            label_counter.update({label:1})
        return l

    for split_name in ['train','dev','test']:
        dataset = "%s.txt.tmp"%split_name
        with open("%s.txt"%split_name,'w') as f:
            f.writelines("%s\n"%count_and_return(l) for l in tqdm(read_and_preprocess(dataset)))

    with open('labels.txt','w') as f:
        f.writelines("%s\n"%l for l in label_counter.keys())

