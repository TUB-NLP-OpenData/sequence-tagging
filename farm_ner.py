"""
based on : FARM/examples/ner.py
"""
from functools import partial

from typing import List, Dict

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

from reading_seqtag_data import TaggedSequence, read_JNLPBA_data, TaggedSeqsDataSet


def build_farm_data(data: List[TaggedSequence]):
    """
    farm wants it like this: {'text': 'Ereignis und Erzählung oder :', 'ner_label': ['O', 'O', 'O', 'O', 'O']}
    the text-field is build by `"text": " ".join(sentence)` see utils.py line 141 in FARM repo
    """

    def _build_dict(tseq: TaggedSequence):
        tokens, tags = zip(*tseq)
        return {"text": " ".join(tokens), "ner_label": tags}

    return [_build_dict(datum) for datum in data]


def ner(data_dicts: Dict[str, List[Dict]]):
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
    ner_labels = build_ner_labels(data_dicts["train_dicts"])

    processor = NERProcessor(
        tokenizer=tokenizer,
        max_seq_len=128,
        data_dir=None,  # noqa
        metric="seq_f1",
        label_list=ner_labels,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets

    data_silo = DataSilo(
        processor=processor, batch_size=batch_size, automatic_loading=False
    )
    data_silo._load_data(**data_dicts)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => NER
    prediction_head = TokenClassificationHead(num_labels=len(ner_labels))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=1e-5,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device,
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
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
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    # save_dir = "saved_models/bert-german-ner-tutorial"
    # model.save(save_dir)
    # processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    # basic_texts = [
    #     {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
    #     {"text": "Martin Müller spielt Handball in Berlin"},
    # ]
    # # model = Inferencer.load(save_dir)
    # result = model.inference_from_dicts(dicts=basic_texts)
    # print(result)


def build_ner_labels(data: List[Dict]):
    return list(set(tag for datum in data for tag in datum["ner_label"]))


def build_fard_data_dicts(dataset: TaggedSeqsDataSet):
    return {
        "%s_dicts" % split_name: build_farm_data(split_data)
        for split_name, split_data in dataset._asdict().items()
    }


if __name__ == "__main__":
    data_supplier = partial(
        read_JNLPBA_data, path=os.environ["HOME"] + "/hpc/scibert/data/ner/JNLPBA"
    )
    dataset = data_supplier()

    ner(build_fard_data_dicts(dataset))
    print()
