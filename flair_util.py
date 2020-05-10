from typing import List, Tuple
from flair.data import Sentence, Token, Corpus
from reading_seqtag_data import TaggedSequence


def build_flair_sentences_from_sequences(
    sequences: List[TaggedSequence],TAG_TYPE="ner"
) -> List[Sentence]:

    sentences = []
    for seq in sequences:
        sentence: Sentence = Sentence()
        [sentence.add_token(Token(tok)) for tok, tag in seq]
        [
            flair_token.add_tag(TAG_TYPE, tag)
            for (token, tag), flair_token in zip(seq, sentence)
        ]
        sentences.append(sentence)
    return sentences


def build_tag_dict(sequences: List[TaggedSequence], tag_type):
    sentences = build_flair_sentences_from_sequences(sequences)
    corpus = Corpus(train=sentences, dev=[], test=[])
    return corpus.make_tag_dictionary(tag_type)