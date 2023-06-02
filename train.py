# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train GPT2 model series with DP (w/ parameter-efficient approach LoRA when lora_dim > 0)"""
import sys
import logging
from dataclasses import dataclass, field
from typing import Sequence
import copy
import utils
from dp_utils import OpacusDPTrainer

import torch
from torch.utils.data import Dataset

import transformers

# from transformers.training_args import ParallelMode
import datasets

import dp_transformers
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
MAX_LENGTH = 50
PROMPT_DICT = {
    "prompt_input": "{intruction}{input}",
    "prompt_no_input": "{instruction}",
}


@dataclass
class ModelArguments:
    model_name: str = field(
        default="gpt2", metadata={"help": "Model name in HuggingFace, e.g. 'gpt2'"}
    )

    lora_dim: int = field(
        default=0, metadata={"help": "LoRA dimension; 0 means LoRA is disabled"}
    )

    sequence_len: int = field(default=128, metadata={"help": "Model sequence length"})

    lora_dropout: float = field(
        default=0.0, metadata={"help": "Dropout probability for LoRA layers"}
    )

    lora_alpha: int = field(default=32, metadata={"help": "LoRA attention alpha"})


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        list_data_dict = utils.DataProcessor().data

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def train(*args):
    train_args, privacy_args, model_args = args
    transformers.set_seed(train_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_config(
        transformers.GPT2Config(
            vocab_size=22,
            n_positions=MAX_LENGTH,
            n_embd=128,
            n_layer=4,
            n_head=2,
            n_inner=256,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            pad_token_id=19,
            bos_token_id=20,
            eos_token_id=21,
        )
    )
    # config = transformers.GPT2Config(
    #     vocab_size=22,
    #     n_positions=MAX_LENGTH,
    #     n_embd=256,
    #     n_layer=4,
    #     n_head=2,
    #     n_inner=512,
    #     activation_function="gelu_new",
    #     resid_pdrop=0.0,
    #     embd_pdrop=0.0,
    #     attn_pdrop=0.0,
    #     layer_norm_epsilon=1e-5,
    #     initializer_range=0.02,
    #     use_cache=True,
    #     pad_token_id=19,
    #     bos_token_id=20,
    #     eos_token_id=21,
    # )
    # model = transformers.GPT2LMHeadModel(config)

    # Load tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast(
        model_max_length=MAX_LENGTH,
        padding_side="right",
        tokenizer_file="tokenizers/star2000_tokenizer.json",
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name)
    # tokenizer.pad_token = -100  # Set a dummy pad token we don't use it anyway

    # Load training dataset
    train_dataset = SupervisedDataset(tokenizer=tokenizer)

    if model_args.lora_dim > 0:
        model = convert_gpt2_attention_to_lora(
            model,
            r=model_args.lora_dim,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            enable_lora=[True, False, True],
            merge_weights=False,
        )
        mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(
            f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}"
        )
        logger.info(
            f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}"
        )

    model = model.cuda()
    model.train()

    if model_args.lora_dim > 0:
        from dp_transformers.grad_sample.lora import lora_layer
    else:
        from dp_transformers.grad_sample.transformers import conv_1d

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(
        tokenizer
    )

    trainer = OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        privacy_args=privacy_args,
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({"final_epsilon_prv": eps_prv, "final_epsilon_rdp": eps_rdp})


def continue_train(*args):
    # Cannot continue training because opacus account state cannot resume from checkpoint
    pass


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (
            dp_transformers.TrainingArguments,
            dp_transformers.PrivacyArguments,
            ModelArguments,
        )
    )
    train(*arg_parser.parse_args_into_dataclasses())
