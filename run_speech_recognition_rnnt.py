#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning NVIDIA RNN-T models for speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import logging
import os
import re
import sys
from dataclasses import dataclass, field

import torchaudio
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from typing import Optional, Dict, Union, List, Any

import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf, open_dict
from models import RNNTBPEModel
from nemo.core import adapter_mixins
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

import datasets
from datasets import DatasetDict, load_dataset, load_metric
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from process_asr_text_tokenizer import __process_data as nemo_process_data, \
    __build_document_from_manifests as nemo_build_document_from_manifests


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    config_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from NVIDIA NeMo NGC."}
    )
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local pretrained model or model identifier."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co or NVIDIA NeMo NGC."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    manifest_path: str = field(
        default="data",
        metadata={
            "help": "Manifest path."
        },
    )
    tokenizer_path: str = field(
        default="tokenizers",
        metadata={
            "help": "Tokenizer path."
        },
    )
    vocab_size: int = field(
        default=1024,
        metadata={"help": "Tokenizer vocab size."}
    )
    tokenizer_type: str = field(
        default="spe",
        metadata={
            "help": "Can be either spe or wpe. spe refers to the Google sentencepiece library tokenizer."
                    "wpe refers to the HuggingFace BERT Word Piece tokenizer."
        },
    )
    spe_type: str = field(
        default="bpe",
        metadata={
            "help": "Type of the SentencePiece model. Can be `bpe`, `unigram`, `char` or `word`."
                    "Used only if `tokenizer_type` == `spe`"
        },
    )
    cutoff_freq: str = field(
        default=0.001,
        metadata={"help": "Drop the least frequent chars from the train set when building the tokenizer."}
    )
    fuse_loss_wer: bool = field(
        default=True,
        metadata={
            "help": "Whether to fuse the computation of prediction net + joint net + loss + WER calculation to be run "
                    "on sub-batches of size `fused_batch_size`"
        }
    )
    fused_batch_size: int = field(
        default=8,
        metadata={
            "help": "`fused_batch_size` is the actual batch size of the prediction net, joint net and transducer loss."
                    "Using small values here will preserve a lot of memory during training, but will make training slower as well."
                    "An optimal ratio of fused_batch_size : per_device_train_batch_size is 1:1."
                    "However, to preserve memory, this ratio can be 1:8 or even 1:16."
        }
    )
    final_decoding_strategy: str = field(
        default="greedy_batch",
        metadata={
            "help": "Decoding strategy for final eval/prediction steps. One of: [`greedy`, `greedy_batch`, `beam`, "
                    "`tsd`, `alsd`]."
        }
    )
    final_num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams for final eval/prediction steps. Increase beam size for better scores, "
                    "but it will take much longer for transcription!"
        }
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze the acoustic encoder of the model. Recommend when fine-tuning on small datasets."}
    )
    unfreeze_encoder: bool = field(
        default=False,
        metadata={"help": "Unfreeze the acoustic encoder of the model after first evaluation step."}
    )
    add_adapter: bool = field(
        default=False,
        metadata={"help": "Add an adapter layer to the encoder of the model."}
    )
    use_adam8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes 8bit AdamW optimiser."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to cache directory for saving and loading datasets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Truncate training audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    max_eval_duration_in_seconds: float = field(
        default=None,
        metadata={
            "help": "Truncate eval/test audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_target_length: Optional[int] = field(
        default=2,
        metadata={
            "help": "The minimum total sequence length for target text after tokenization. Sequences shorter "
                    "than this will be filtered."
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
                    "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
                    "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
                    "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={"help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"},
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    wandb_project: str = field(
        default="speech-recognition-rnnt",
        metadata={"help": "The name of the wandb project."},
    )
    ignore_verifications: bool = field(
        default=False,
        metadata={
            "help": "Ignore the verifications of the downloaded/processed dataset information in `load_dataset` (checksums/size/splits/...)."
        }
    )
    torchaudio_resampler: bool = field(
        default=False,
        metadata={
            "help": "Whether to use torchaudio to resample. If `False` (default) will use the default datataset backed."
        }
    )


def write_wandb_pred(pred_str, label_str, prefix="eval"):
    # convert str data to a wandb compatible format
    str_data = [[label_str[i], pred_str[i]] for i in range(len(pred_str))]
    # we'll log all predictions for the last epoch
    wandb.log(
        {
            f"{prefix}/predictions": wandb.Table(
                columns=["label_str", "pred_str"], data=str_data
            )
        },
    )


def build_tokenizer(model_args, data_args, manifests):
    """
    Function to build a NeMo tokenizer from manifest file(s).
    Copied from https://github.com/NVIDIA/NeMo/blob/66c7677cd4a68d78965d4905dd1febbf5385dff3/scripts/tokenizers/process_asr_text_tokenizer.py#L268
    """
    data_root = model_args.tokenizer_path
    if isinstance(manifests, list):
        joint_manifests = ",".join(manifests)
    else:
        joint_manifests = manifests
    vocab_size = model_args.vocab_size
    tokenizer = model_args.tokenizer_type
    spe_type = model_args.spe_type
    if not 0 <= model_args.cutoff_freq < 1:
        raise ValueError(f"`cutoff_freq` must be between zero and one, got {model_args.cutoff_freq}")
    spe_character_coverage = 1 - model_args.cutoff_freq

    logger.info("Building tokenizer...")
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    text_corpus_path = nemo_build_document_from_manifests(data_root, joint_manifests)

    tokenizer_path = nemo_process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=data_args.do_lower_case,
        spe_character_coverage=spe_character_coverage,
        spe_sample_size=-1,
        spe_train_extremely_large_corpus=False,
        spe_max_sentencepiece_length=-1,
        spe_bos=False,
        spe_eos=False,
        spe_pad=False,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logger.info('Done!')

    # Tokenizer path
    if tokenizer == 'spe':
        tokenizer_dir = os.path.join(data_root, f"tokenizer_spe_{spe_type}_v{vocab_size}")
        tokenizer_type_cfg = "bpe"
    else:
        tokenizer_dir = os.path.join(data_root, f"tokenizer_wpe_v{vocab_size}")
        tokenizer_type_cfg = "wpe"

    return tokenizer_dir, tokenizer_type_cfg


def NeMoDataCollator(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Data collator that will dynamically pad the inputs received.
    Since NeMo models don't have a HF processor defined (feature extractor + tokenizer), we'll pad by hand...
    The padding idx is arbitrary: we provide the model with the input lengths and label lengths, from which
    all the relevant padding information is inferred. Thus, we'll use the default np.pad padding idx (0).
    """
    # split inputs and labels since they have to be of different lengths
    # and need different padding methods
    input_ids = [feature["input_ids"] for feature in features]
    labels = [feature["labels"] for feature in features]

    # first, pad the audio inputs to max_len
    input_lengths = [feature["input_lengths"] for feature in features]
    max_input_len = max(input_lengths)
    input_ids = [np.pad(input_val, (0, max_input_len - input_len), 'constant') for input_val, input_len in
                 zip(input_ids, input_lengths)]

    # next, pad the target labels to max_len
    label_lengths = [len(lab) for lab in labels]
    max_label_len = max(label_lengths)
    labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant') for lab, lab_len in zip(labels, label_lengths)]

    batch = {"input_lengths": input_lengths, "labels": labels, "label_lengths": label_lengths}

    # return batch as a pt tensor (list -> np.array -> torch.tensor)
    batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

    # leave all ints as are, convert float64 to pt float
    batch["input_ids"] = torch.tensor(np.array(input_ids, dtype=np.float32), requires_grad=False)

    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set wandb project ID before instantiating the Trainer
    os.environ["WANDB_PROJECT"] = data_args.wandb_project

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load the model config (discarding optimiser and trainer attributes)
    config = OmegaConf.load(model_args.config_path).model

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_verifications=data_args.ignore_verifications,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_verifications=data_args.ignore_verifications,
        )

    if training_args.do_predict:
        test_split = data_args.test_split_name.split("+")
        for split in test_split:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_verifications=data_args.ignore_verifications,
            )

    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        raise ValueError(
            "Cannot not train, not do evaluation and not do prediction. At least one of "
            "training, evaluation or prediction has to be done."
        )

    # if not training, there is no need to run multiple epochs
    if not training_args.do_train:
        training_args.num_train_epochs = 1

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 6. Resample speech dataset ALWAYS
    if data_args.torchaudio_resampler:
        # TODO: remove hardcoding of orig sr
        resampler = torchaudio.transforms.Resample(8_000, config.sample_rate)
    else:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=config.sample_rate)
        )
        resampler = None

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * config.sample_rate)
    min_input_length = max(int(data_args.min_duration_in_seconds * config.sample_rate), 1)
    max_eval_input_length = int(data_args.max_eval_duration_in_seconds * config.sample_rate) if data_args.max_eval_duration_in_seconds else None
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    do_lower_case = data_args.do_lower_case
    dataset_name = data_args.dataset_name

    # Define tokens to ignore/replace
    tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]
    gigaspeech_punctuation = {" <comma>": ",", " <period>": ".", " <questionmark>": "?", " <exclamationpoint>": "!"}
    gigaspeech_disfluencies = ["<other>", "<sil>"]
    swb_disfluencies = ["[noise]", "[laughter]", "[silence]", "[vocalized-noise]", "<a_aside>", "<b_aside>", "<e_aside>",
                        "[laughter-", "_1", "[laugh]", "[sigh]", "[cough]", "[mn]", "[breath]", "[lipsmack]",
                        "[sneeze]", "[skip]", "[pause]", "(%hesitation)", "(%HESITATION)"]
    swb_punctuations = ["{", "}", "[", "]-", "]", "((", "))", "(", ")"]
    swb_fillers = r"\b(uh|uhm|um|hmm|mm|mhm|mmm)\b"
    earnings_disfluencies = ["<noise>", "<crosstalk>", "<affirmative>", "<inaudible>", "inaudible", "<laugh>", "<silence>"]
    ignore_segments = ["ignore_time_segment_in_scoring", "<noise>", "<music>", "[noise]", "[laughter]", "[silence]",
                       "[vocalized-noise]", "<crosstalk>", "<affirmative>", "<inaudible>", "<laugh>", ""]
    ignore_segments = ignore_segments + gigaspeech_disfluencies + swb_disfluencies + earnings_disfluencies

    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval and data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    if training_args.do_predict and data_args.max_predict_samples is not None:
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_predict_samples))

    # filter data where the targets are ignored in scoring
    def is_target_labels(input_str):
        return input_str.lower() not in ignore_segments

    raw_datasets = raw_datasets.filter(
        is_target_labels,
        num_proc=num_workers,
        input_columns=[text_column_name],
        desc="filtering data where the targets are ignored in scoring",
    )

    def prepare_dataset(batch):
        # pre-process audio
        try:
            sample = batch[audio_column_name]
        except ValueError:
            # E22: some samples are empty (no audio). Reading the empty audio array will trigger
            # a soundfile ValueError. For now, we'll manually set these arrays to a zero array.
            # They will be filtered in the subsequent filtering stage and so are
            # explicitly ignored during training.
            sample = {"array": np.array([0.]), "sampling_rate": config.sample_rate}

        if resampler is not None:
            speech_tensor = torch.FloatTensor(sample["array"])
            speech_tensor = speech_tensor.squeeze()
            speech_tensor = resampler(speech_tensor)
            sample["array"] = speech_tensor.numpy()
            sample["sampling_rate"] = resampler.new_freq

        # NeMo RNNT model performs the audio preprocessing in the `.forward()` call
        # => we only need to supply it with the raw audio values
        batch["input_ids"] = sample["array"]
        batch["input_lengths"] = len(sample["array"])

        # 'Error correction' of targets
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]

        # LibriSpeech ASR
        if dataset_name == "librispeech_asr":
            pass  # no error correction necessary

        # VoxPopuli
        if dataset_name == "google/xtreme_s":
            pass  # no error correction necessary

        # Common Voice 9
        if dataset_name == "mozilla-foundation/common_voice_9_0":
            if input_str.startswith('"') and input_str.endswith('"'):
                # we can remove trailing quotation marks as they do not affect the transcription
                input_str = input_str[1:-1]
            # replace double quotation marks with single
            input_str = input_str.replace('""', '"')

        # TED-LIUM (Release 3)
        if dataset_name == "LIUM/tedlium":
            # delete the <unk> token from the text
            input_str = input_str.replace("<unk>", "")
            # replace spaced apostrophes with un-spaced (it 's -> it's)
            for contraction in tedlium_contractions:
                input_str = input_str.replace(contraction, contraction[1:])

        # GigaSpeech
        if dataset_name == "speechcolab/gigaspeech":
            for disfluency in gigaspeech_disfluencies:
                input_str = input_str.replace(disfluency, "")
            # convert spelled out punctuation to symbolic form
            for punctuation, replacement in gigaspeech_punctuation.items():
                input_str = input_str.replace(punctuation, replacement)

        # SWB: hide the path to the private HF dataset
        if "switchboard" in dataset_name:
            # In one conversation people speak some German phrases that are tagged as
            # <german (( ja wohl )) > -- we remove these
            input_str = re.sub("<[^>]*>", "", input_str)

            # Remove junk tokens
            for disfluency in swb_disfluencies:
                input_str = input_str.replace(disfluency, "")

            # normalise acronyms (Fisher: u_.c_.l_.a., SWBD: u c l a)
            input_str = input_str.replace("_.", " ").replace(".", "")
            # Replace partially pronounced words (square brackets + hyphen): westmin[ster]- to westmin- or -[go]ing to -ing
            # Replace anomalous words (square brackets + backslack): [lemguini/linguini] to linguini
            # Replace the combo of the two: [lem[guini]-/linguini] to lem-
            # Example: we [ah/are] -[go]ing to westmin[ster]- for [lem[guini]-/linguini]
            # Target: we ah -ing to westmin- for lem-
            # Treat anomalous words first then destroy the content of all square brackets (partially pronounced words)

            # First treat partially pronounced anomalous words by removing correct word: [lem[guini]-/linguini] to [lem[guini]-
            input_str = re.sub(r"\-\/.*?\]", "-", input_str)

            # Now replace anomalous words with their correct transcriptions: [lemguini/linguini] to linguini
            split_str = input_str.split("/")
            if len(split_str) > 1:
                input_str = " ".join(
                    [" ".join([" ".join(i.split(" ")[:-1]) for i in split_str])] + [split_str[-1].split(" ")[-1]])

            # Remove the trailing brackets on the start/end of words
            processed_str = []
            for word in input_str.split():
                if word[0] == "[":
                    processed_str.append(word[1:])
                elif word[-1] == "]":
                    processed_str.append(word[:-1])
                else:
                    processed_str.append(word)

            # Stick the processed words back together
            input_str = " ".join(processed_str)

            # Now we can remove all words in square brackets: -[go]ing to -ing
            input_str = re.sub(r"\-\[(.*?)\]", "-", input_str)

            # westmin[ster]- to westmin-
            input_str = re.sub(r"\[(.*?)\]\-", "-", input_str)

            # tech[n]ology to tech-ology
            input_str = re.sub(r"\[(.*?)\]", "-", input_str)

            # partially pronounced words are now done!
            # remove erroneous punctuations (curly braces, trailing square brackets, etc.)
            for punctuation in swb_punctuations:
                input_str = input_str.replace(punctuation, "")

            # Remove fillers from the train set not present in the test set
            input_str = re.sub(swb_fillers, "", input_str)

        # Earnings 22: still figuring out best segmenting method. Thus, dataset name subject to change
        if "earnings22" in dataset_name:
            # Remove the 100ms offset at the end of the sample
            sampling_rate = sample["sampling_rate"]
            offset = int(100 * (10 ** -3) * sampling_rate)
            batch["input_ids"] = sample["array"][:-offset]
            batch["input_lengths"] = len(batch["input_ids"])
            # Remove  junk tokens
            for disfluency in earnings_disfluencies:
                input_str = input_str.replace(disfluency, "")

        # SPGISpeech
        if dataset_name == "kensho/spgispeech":
            pass  # no error correction necessary

        # JIWER compliance (for WER/CER calc.)
        # remove multiple spaces
        input_str = re.sub(r"\s\s+", " ", input_str)
        # strip trailing spaces
        input_str = input_str.strip()

        # We can't currently tokenize the dataset... we need the pre-processed text data in order to
        # build our SPE tokenizer. Once we've defined our tokenizer, we can come back and
        # tokenize the text. For now, just return the pre-processed text data
        batch["processed_text"] = input_str
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter training data with inputs shorter than min_input_length or longer than max_input_length
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_lengths"],
        )

    if max_eval_input_length is not None:
        # filter training data with inputs longer than max_input_length
        def is_eval_audio_in_length_range(length):
            return min_input_length < length < max_eval_input_length

        vectorized_datasets = vectorized_datasets.filter(
            is_eval_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_lengths"],
        )

    def is_labels_non_zero(transcription):
        return len(transcription) > 0

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_non_zero,
        num_proc=num_workers,
        input_columns=["processed_text"],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # Function to build a NeMo tokenizer manifest from a HF dataset
    # TODO: with a bit of hacking around we can probably bypass this step entirely
    def build_manifest(ds, manifest_path):
        with open(manifest_path, 'w') as fout:
            for sample in tqdm(ds["processed_text"]):
                # Write the metadata to the manifest
                metadata = {
                    "text": sample
                }
                json.dump(metadata, fout)
                fout.write('\n')

    config.train_ds = config.validation_ds = config.test_ds = None

    if not os.path.exists(model_args.manifest_path) and training_args.do_train:
        os.makedirs(model_args.manifest_path)
        manifest = os.path.join(model_args.manifest_path, "train.json")
        logger.info(f"Building training manifest at {manifest}")
        build_manifest(vectorized_datasets["train"], manifest)
    else:
        manifest = os.path.join(model_args.manifest_path, "train.json")
        logger.info(f"Re-using training manifest at {manifest}")

    tokenizer_dir, tokenizer_type_cfg = build_tokenizer(model_args, data_args, manifest)

    # generalise the script later to load a pre-built tokenizer for eval only
    config.tokenizer.dir = tokenizer_dir
    config.tokenizer.type = tokenizer_type_cfg

    if model_args.add_adapter:
        # Utility method to check and update the model config
        def update_model_config_to_support_adapter(model_cfg):
            with open_dict(model_cfg):
                adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
                if adapter_metadata is not None:
                    model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

            logging.info("Updated encoder _target_ model :", model_cfg.encoder._target_)
            return model_cfg

        config = update_model_config_to_support_adapter(config)

    # possibly fused-computation of prediction net + joint net + loss + WER calculation
    config.joint.fuse_loss_wer = model_args.fuse_loss_wer
    if model_args.fuse_loss_wer:
        config.joint.fused_batch_size = model_args.fused_batch_size

    if model_args.model_name_or_path is not None:
        # load pre-trained model weights
        model = RNNTBPEModel.from_pretrained(model_args.model_name_or_path, override_config_path=config, map_location="cpu")
        model.save_name = model_args.model_name_or_path

        pretrained_decoder = model.decoder.state_dict()
        pretrained_joint = model.joint.state_dict()
        model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type_cfg)

        # TODO: add checks for loading decoder/joint state dict
        model.decoder.load_state_dict(pretrained_decoder)
        model.joint.load_state_dict(pretrained_joint)

    elif model_args.pretrained_model_name_or_path is not None:
        model = RNNTBPEModel.restore_from(model_args.pretrained_model_name_or_path, override_config_path=config,
                                             map_location="cpu")
        model.save_name = model_args.config_path.split("/")[-1].split(".")[0]

    else:
        model = RNNTBPEModel(cfg=config)
        model.save_name = model_args.config_path.split("/")[-1].split(".")[0]
        model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type_cfg)

    if model_args.add_adapter:
        adapter_name = model_args.config_path.split("/")[-1].split(".")[0]
        adapter_dim = model.cfg.encoder.d_model
        adapter_activation = "swish"
        adapter_norm_position = "post"
        adapter_cfg = LinearAdapterConfig(
            in_features=model.cfg.encoder.d_model,
            # conformer specific model dim. Every layer emits this dim at its output.
            dim=adapter_dim,  # the bottleneck dimension of the adapter
            activation=adapter_activation,  # activation used in bottleneck block
            norm_position=adapter_norm_position,  # whether to use LayerNorm at the beginning or the end of the adapter
        )
        logger.info("Adapter config: ", adapter_cfg)
        model.add_adapter(name=adapter_name, cfg=adapter_cfg)
        model.set_enabled_adapters(enabled=False)  # disable all adapters
        model.set_enabled_adapters(name=adapter_name, enabled=True)  # enable only the current adapter we want to train

    def enable_bn(m):
        if type(m) == nn.BatchNorm1d:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

    if model_args.freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")

    if model_args.add_adapter:
        model.unfreeze_enabled_adapters()
        logging.info("Model adapter has been unfrozen")

    # now that we have our model and tokenizer defined, we can tokenize the text data
    tokenizer = model.tokenizer.tokenizer.encode_as_ids

    def tokenize_transcripts(batch):
        batch["labels"] = tokenizer(batch["processed_text"])
        return batch

    vectorized_datasets = vectorized_datasets.map(tokenize_transcripts, num_proc=num_workers,
                                                  desc="Tokenizing datasets...",)

    def compute_metrics(pred):
        # Tuple of WERs returned by the model during eval: (wer, wer_num, wer_denom)
        wer_num = pred.predictions[1]
        wer_denom = pred.predictions[2]
        # compute WERs over concat batches
        wer = sum(wer_num) / sum(wer_denom)
        return {"wer": wer}

    class UnfreezeEncoderCallback(TrainerCallback):
        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            model.encoder.unfreeze()
            print("Model encoder has been unfrozen")

    class NeMoTrainer(Trainer):
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            self.model.save_to(save_path=os.path.join(output_dir, model.save_name + ".nemo"))
            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        def transcribe(self, test_dataset: Dataset) -> List[Any]:
            self.model.eval()
            test_dataloader = self.get_test_dataloader(test_dataset)
            hypotheses = []
            for test_batch in tqdm(test_dataloader, desc="Transcribing"):
                inputs = self._prepare_inputs(test_batch)
                best_hyp, all_hyp = self.model.transcribe(**inputs)
                hypotheses += best_hyp
                del test_batch
            return hypotheses


    # Initialize Trainer
    trainer = NeMoTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets['train'] if training_args.do_train else None,
        eval_dataset=vectorized_datasets['eval'] if training_args.do_eval else None,
        data_collator=NeMoDataCollator,
        callbacks=[UnfreezeEncoderCallback] if model_args.unfreeze_encoder else None,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Change decoding strategy for final eval/predict
    if training_args.do_eval or training_args.do_predict:
        # set beam search decoding config
        beam_decoding_config = copy.deepcopy(trainer.model.cfg.decoding)
        beam_decoding_config.strategy = model_args.final_decoding_strategy
        beam_decoding_config.beam.beam_size = model_args.final_num_beams

        trainer.model.change_decoding_strategy(beam_decoding_config)

    results = {}
    if training_args.do_eval:
        logger.info(f"*** Running Final Evaluation ({model_args.final_decoding_strategy}) ***")

        predictions = trainer.transcribe(vectorized_datasets["eval"])
        targets = model.tokenizer.ids_to_text(vectorized_datasets["eval"]["labels"])

        cer_metric = load_metric("cer")
        wer_metric = load_metric("wer")

        cer = cer_metric.compute(predictions=predictions, references=targets)
        wer = wer_metric.compute(predictions=predictions, references=targets)

        metrics = {f"eval_cer": cer, f"eval_wer": wer}

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if "wandb" in training_args.report_to:
            if not training_args.do_train:
                wandb.init(name=training_args.run_name, project=data_args.wandb_project)
            metrics = {os.path.join("eval", k[len("eval") + 1:]): v for k, v in metrics.items()}
            # wandb.init(project=data_args.wandb_project, name=training_args.run_name)
            wandb.log(metrics)
            write_wandb_pred(predictions, targets, prefix="eval")

    if training_args.do_predict:
        logger.info(f"*** Running Final Prediction ({model_args.final_decoding_strategy}) ***")

        for split in test_split:
            predictions = trainer.transcribe(vectorized_datasets[split])
            targets = model.tokenizer.ids_to_text(vectorized_datasets[split]["labels"])

            cer_metric = load_metric("cer")
            wer_metric = load_metric("wer")

            cer = cer_metric.compute(predictions=predictions, references=targets)
            wer = wer_metric.compute(predictions=predictions, references=targets)

            metrics = {f"{split}_cer": cer, f"{split}_wer": wer}

            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                    vectorized_datasets[split])
            )
            metrics[f"{split}_samples"] = min(max_predict_samples, len(vectorized_datasets[split]))

            trainer.log_metrics(split, metrics)
            trainer.save_metrics(split, metrics)

            if "wandb" in training_args.report_to:
                metrics = {os.path.join(split, k[len(split) + 1:]): v for k, v in metrics.items()}
                wandb.log(metrics)
                write_wandb_pred(predictions, targets, prefix=split)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    #else:
        #trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
