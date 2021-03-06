"""
based on : FARM/examples/ner.py
"""
import logging
import os
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from functools import partial
from pprint import pprint
from time import time
from typing import List, Dict, Any, Tuple, NamedTuple

from data_splitting import build_data_supplier_splits_trainset_only, EvalJob
from experiment_util import SeqTagScoreTask, SeqTagTaskData, Splits
from mlutil.crossvalidation import calc_mean_std_scores
from reading_seqtag_data import TaggedSequence, TaggedSeqsDataSet, read_conll03_en
from seq_tag_util import Sequences, bilou2bio

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING,
)
# logging.disable(logging.CRITICAL)
# logger = logging.getLogger() #TODO(tilo): this is still not working!
# logger.setLevel(logging.WARNING)


class TokenClassificationHeadPredictSequence(TokenClassificationHead):
    def formatted_preds(
        self, logits, initial_mask, samples, return_class_probs=False, **kwargs
    ):
        return self.logits_to_preds(logits, initial_mask)


def build_farm_data(data: List[TaggedSequence]):
    """
    farm wants it like this: {'text': 'Ereignis und Erzählung oder :', 'ner_label': ['O', 'O', 'O', 'O', 'O']}
    the text-field is build by `"text": " ".join(sentence)` see utils.py line 141 in FARM repo
    """

    def _build_dict(tseq: TaggedSequence):
        tokens, tags = zip(*tseq)
        return {"text": " ".join(tokens), "ner_label": list(tags)}

    return [_build_dict(datum) for datum in data]


def build_farm_data_dicts(dataset):
    return {
        split_name: build_farm_data(split_data)
        for split_name, split_data in dataset.items()
    }


NIT = "X"  # non initial token


class Params(NamedTuple):
    batch_size: int=32
    n_epochs: int = 5


def build_data_silo(params: Params, splits, processor):
    data_silo = DataSilo(
        processor=processor,
        batch_size=params.batch_size,
        automatic_loading=False,
        max_processes=4,
    )
    farm_data = build_farm_data_dicts(splits)
    data_silo._load_data(
        **{"%s_dicts" % split_name: d for split_name, d in farm_data.items()}
    )
    return data_silo,farm_data


def build_model(n_batches, device, n_epochs, task_data):
    language_model = LanguageModel.load(task_data["lang_model"])
    prediction_head = TokenClassificationHeadPredictSequence(
        num_labels=task_data["num_labels"]
    )
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
        n_batches=n_batches,
        n_epochs=n_epochs,
        device=device,
    )
    return lr_schedule, model, optimizer


def predict_iob(inferencer, sn, dicts):
    batches = inferencer.inference_from_dicts(dicts=dicts)
    prediction = [
        bilou2bio([t if t != NIT else "O" for t in seq])
        for batch in batches
        for seq in batch
    ]
    targets = [bilou2bio(d["ner_label"]) for d in dicts]
    pred_target = [(p, t) for (p, t) in zip(prediction, targets) if len(p) == len(t)]
    print(
        "WARNING: %s got %d invalid predictions" % (sn, len(targets) - len(pred_target))
    )
    prediction, targets = [list(x) for x in zip(*pred_target)]

    assert all([len(t) == len(p) for t, p in zip(targets, prediction)])
    assert len(targets) == len(prediction)
    return prediction, targets


class FarmSeqTagScoreTask(SeqTagScoreTask):
    @classmethod
    def predict_with_targets(
        cls, splits: Splits, task_data: Dict[str, Any]
    ) -> Dict[str, Tuple[Sequences, Sequences]]:
        params: Params = task_data["params"]
        data_silo,farm_data = build_data_silo(params, splits, task_data["processor"])

        set_all_seeds(seed=42)
        device, n_gpu = initialize_device_settings(use_cuda=True)

        evaluate_every = 400

        lr_schedule, model, optimizer = build_model(
            len(data_silo.loaders["train"]), device, params.n_epochs, task_data
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=params.n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
        )

        trainer.train()

        # 8. Hooray! You have a model. Store it:
        # save_dir = "saved_models/bert-german-ner-tutorial"
        # model.save(save_dir)
        # processor.save(save_dir)

        inferencer = Inferencer(
            model,
            task_data["processor"],
            task_type="ner",
            batch_size=16,
            num_processes=8,
            gpu=True,
        )

        out = {
            split_name: predict_iob(inferencer, split_name, split_data)
            for split_name, split_data in farm_data.items()
        }

        return out

    @staticmethod
    def build_task_data(params: Params, data_supplier) -> SeqTagTaskData:
        dataset_dict: Dict[str, List[TaggedSequence]] = data_supplier()
        ner_labels = ["[PAD]", NIT] + list(
            set(
                tag
                for taggedseqs in dataset_dict.values()
                for taggedseq in taggedseqs
                for tok, tag in taggedseq
            )
        )

        ml_logger = MLFlowLogger(
            tracking_uri=os.environ["HOME"] + "/data/mlflow_experiments/mlruns"
        )
        ml_logger.init_experiment(
            experiment_name="Sequence_Tagging", run_name="Run_ner"
        )

        lang_model = "bert-base-cased"
        do_lower_case = False

        tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
        )

        processor = NERProcessor(
            tokenizer=tokenizer,
            max_seq_len=128,
            data_dir=None,  # noqa
            metric="seq_f1",
            label_list=ner_labels,
        )

        task_data = {
            "num_labels":len(ner_labels),
            "lang_model": lang_model,
            "ml_logger": ml_logger,
            "processor": processor,
            "params": params,
        }
        return SeqTagTaskData(data=dataset_dict, task_data=task_data)


if __name__ == "__main__":
    import os

    raw_data_supplier = partial(
        read_conll03_en, path=os.environ["HOME"] + "/data/IE/seqtag_data"
    )

    num_folds = 1
    data_supplier, splits = build_data_supplier_splits_trainset_only(
        raw_data_supplier, num_folds, 0.2
    )

    start = time()
    task = FarmSeqTagScoreTask(
        params=Params(batch_size=32, n_epochs=4), data_supplier=data_supplier
    )
    num_workers = 0  # min(multiprocessing.cpu_count() - 1, num_folds)
    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=num_workers)
    print(
        " %d folds %d workers took: %0.2f seconds"
        % (num_folds, num_workers, time() - start)
    )
    pprint(m_scores_std_scores)
