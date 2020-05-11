"""
based on : FARM/examples/ner.py
"""
from pprint import pprint

from time import time

from functools import partial

from typing import List, Dict, Any, Tuple

import logging
import os
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

from eval_jobs import EvalJob, shufflesplit_trainset_only
from experiment_util import SeqTagScoreTask
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import TaggedSequence, read_JNLPBA_data, TaggedSeqsDataSet
from seq_tag_util import Sequences


class TokenClassificationHeadPredictSequence(TokenClassificationHead):
    def formatted_preds(
        self, logits, initial_mask, samples, return_class_probs=False, **kwargs
    ):
        # res = {"task": "ner", "predictions": }
        return self.logits_to_preds(logits, initial_mask)


def build_farm_data(data: List[TaggedSequence]):
    """
    farm wants it like this: {'text': 'Ereignis und Erzählung oder :', 'ner_label': ['O', 'O', 'O', 'O', 'O']}
    the text-field is build by `"text": " ".join(sentence)` see utils.py line 141 in FARM repo
    """

    def _build_dict(tseq: TaggedSequence):
        tokens, tags = zip(*tseq)
        return {"text": " ".join(tokens), "ner_label": tags}

    return [_build_dict(datum) for datum in data[:100]]


def ner(data_dicts: Dict[str, List[Dict]], ner_labels, params={}):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(
        tracking_uri=os.environ["HOME"] + "/data/mlflow_experiments/mlruns"
    )
    ml_logger.init_experiment(experiment_name="Sequence_Tagging", run_name="Run_ner")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 4
    batch_size = 32
    evaluate_every = 400
    lang_model = "bert-base-cased"
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # See test/sample/ner/train-sample.txt for an example of the data format that is expected by the Processor
    # fmt: off
    # ner_labels = ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"]
    # fmt: on

    processor = NERProcessor(
        tokenizer=tokenizer,
        max_seq_len=128,
        data_dir=None,  # noqa
        metric="seq_f1",
        label_list=ner_labels,
    )

    data_silo = DataSilo(
        processor=processor, batch_size=batch_size, automatic_loading=False
    )
    data_silo._load_data(
        **{"%s_dicts" % split_name: d for split_name, d in data_dicts.items()}
    )

    language_model = LanguageModel.load(lang_model)
    prediction_head = TokenClassificationHeadPredictSequence(num_labels=len(ner_labels))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=1e-5,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    # 7. Let it grow
    # trainer.train()

    # 8. Hooray! You have a model. Store it:
    # save_dir = "saved_models/bert-german-ner-tutorial"
    # model.save(save_dir)
    # processor.save(save_dir)

    inferencer = Inferencer(model, processor, task_type="ner", gpu=True, batch_size=16)

    def predict_iob(dicts):
        batches = inferencer.inference_from_dicts(dicts=dicts)
        prediction = [seq for batch in batches for seq in batch]
        targets = [d["ner_label"] for d in dicts]
        return prediction, targets

    return {
        split_name: predict_iob(split_data)
        for split_name, split_data in data_dicts.items()
    }


def build_farm_data_dicts(dataset: TaggedSeqsDataSet):
    return {
        split_name: build_farm_data(split_data)
        for split_name, split_data in dataset._asdict().items()
    }


class FarmSeqTagScoreTask(SeqTagScoreTask):
    @classmethod
    def predict_with_targets(
        cls, job: EvalJob, task_data: Dict[str, Any]
    ) -> Dict[str, Tuple[Sequences, Sequences]]:
        return ner(task_data["data"], task_data["ner_labels"])

    @staticmethod
    def build_task_data(params, data_supplier) -> Dict[str, Any]:
        dataset: TaggedSeqsDataSet = data_supplier()
        dataset_dict: Dict[str, List[TaggedSequence]] = dataset._asdict()
        ner_labels = ["[PAD]", "X"] + list(
            set(
                tag
                for taggedseqs in dataset_dict.values()
                for taggedseq in taggedseqs
                for tok, tag in taggedseq
            )
        )

        data = build_farm_data_dicts(dataset)
        return {
            "data": data,
            "params": params,
            "ner_labels": ner_labels,
        }


if __name__ == "__main__":
    from json import encoder

    encoder.FLOAT_REPR = lambda o: format(o, ".2f")

    data_supplier = partial(
        read_JNLPBA_data, path=os.environ["HOME"] + "/scibert/data/ner/JNLPBA"
    )
    dataset = data_supplier()
    num_folds = 1

    splits = shufflesplit_trainset_only(dataset, num_folds)
    n_jobs = 0  # min(5, num_folds)# needs to be zero if using Transformers

    exp_name = "flair-glove"
    task = FarmSeqTagScoreTask(params={"bla": 1}, data_supplier=data_supplier)
    start = time()
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=n_jobs)
    duration = time() - start
    print(
        "farm-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds"
        % (num_folds, n_jobs, duration)
    )
    exp_results = {
        "scores": m_scores_std_scores,
        "overall-time": duration,
        "num-folds": num_folds,
    }
    pprint(exp_results)
    # data_io.write_json("%s.json" % exp_name, exp_results)
