from collections import namedtuple
from typing import List, Tuple, NamedTuple

import numpy as np
from flair.data import iob2, iob_iobes
from sklearn import metrics
from seqeval.metrics import f1_score

BIO = {"B", "I", "O"}
Sequences = List[List[str]]

def calc_seqtag_f1_scores(
    predictions: Sequences, targets: Sequences,
):
    assert set([t[0] for s in targets for t in s]).issubset(BIO)
    assert set([t[0] for s in predictions for t in s]).issubset(BIO)
    assert all([len(t)==len(p) for t,p in zip(targets,predictions)])
    _, _, f1_train = spanlevel_pr_re_f1(predictions, targets)
    # tokenlevel_scores = calc_seqtag_tokenlevel_scores(targets, predictions)
    return {
        # "token-level": tokenlevel_scores,
        "f1-micro-spanlevel": f1_train,
        "seqeval-f1":f1_score(targets,predictions)
    }


def mark_text(text, char_spans):
    sorted_spans = sorted(char_spans, key=lambda sp: -sp[0])
    for span in sorted_spans:
        assert span[1] > span[0]
        text = text[: span[1]] + "</" + span[2] + ">" + text[span[1] :]
        text = text[: span[0]] + "<" + span[2] + ">" + text[span[0] :]
    return text


def correct_biotags(tag_seq):
    correction_counter = 0
    corr_tag_seq = tag_seq
    for i in range(len(tag_seq)):
        if i > 0 and tag_seq[i - 1] is not "O":
            previous_label = tag_seq[i - 1][2:]
        else:
            previous_label = "O"
        current_label = tag_seq[i][2:]
        if tag_seq[i].startswith("I-") and not current_label is not previous_label:
            correction_counter += 1
            corr_tag_seq[i] = "B-" + current_label
    return corr_tag_seq

def iob2iobes(tags:List[str]):
    Label = namedtuple("Label","value") # just to please flair
    tags = [Label(tag) for tag in tags]
    iob2(tags)
    tags = iob_iobes(tags)
    return tags


def bilou2bio(tag_seq):
    """
    BILOU to BIO
    or
    BIOES to BIO
    E == L
    S == U
    """
    bio_tags = tag_seq
    for i in range(len(tag_seq)):
        if tag_seq[i].startswith("U-") or tag_seq[i].startswith("S-"):
            bio_tags[i] = "B-" + tag_seq[i][2:]
        elif tag_seq[i].startswith("L-") or tag_seq[i].startswith("E-"):
            bio_tags[i] = "I-" + tag_seq[i][2:]
    assert set([t[0] for t in bio_tags]).issubset(BIO),set([t[0] for t in bio_tags])
    return bio_tags


def spanlevel_pr_re_f1(label_pred, label_correct):
    pred_counts = [
        compute_TP_P(pred, gold) for pred, gold in zip(label_pred, label_correct)
    ]
    gold_counts = [
        compute_TP_P(gold, pred) for pred, gold in zip(label_pred, label_correct)
    ]
    prec = np.sum([x[0] for x in pred_counts]) / np.sum([x[1] for x in pred_counts])
    rec = np.sum([x[0] for x in gold_counts]) / np.sum([x[1] for x in gold_counts])
    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def calc_seqtag_tokenlevel_scores(
    gold_seqs: Sequences, pred_seqs: Sequences
):
    gold_flattened = [l for seq in gold_seqs for l in seq]
    pred_flattened = [l for seq in pred_seqs for l in seq]
    assert len(gold_flattened) == len(pred_flattened) and len(gold_flattened) > 0
    label_set = list(set(gold_flattened + pred_flattened))
    scores = {
        "f1-micro": metrics.f1_score(gold_flattened, pred_flattened, average="micro"),
        "f1-macro": metrics.f1_score(gold_flattened, pred_flattened, average="macro"),
        "cohens-kappa": metrics.cohen_kappa_score(gold_flattened, pred_flattened),
        "clf-report": metrics.classification_report(
            gold_flattened,
            pred_flattened,
            target_names=label_set,
            digits=3,
            output_dict=True,
        ),
    }
    return scores


def compute_TP_P(guessed, correct):
    assert len(guessed) == len(correct)
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == "B":  # A new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True

                while (
                    idx < len(guessed) and guessed[idx][0] == "I"
                ):  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == "I":  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    return correctCount, count


def char_precise_spans_to_token_spans(
    char_spans: List[Tuple[int, int, str]], token_spans: List[Tuple[int, int]]
):
    spans = []
    for char_start, char_end, label in char_spans:
        closest_token_start = int(
            np.argmin(
                [np.abs(token_start - char_start) for token_start, _ in token_spans]
            )
        )
        closest_token_end = int(
            np.argmin([np.abs(token_end - char_end) for _, token_end in token_spans])
        )
        spans.append((closest_token_start, closest_token_end, label))
    return spans
