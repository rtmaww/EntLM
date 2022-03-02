import argparse
import logging
import math
import os
import random
import time
import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from tasks import NER

from viterbi import ViterbiDecoder

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    XLNetLMHeadModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# from transformers.utils.versions import require_version
NEAR_0 = 1e-10

logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(        
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default='', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='bert-base-cased'
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--label_schema",
        default='IO',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--eval_label_schema",
        default='BIO',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--label_map_path",
        default=None,
        type=str,
        help="label map path",
    )
    parser.add_argument(
        "--do_crf",
        action="store_true",
    )
    parser.add_argument(
        "--crf_raw_path",
        default=None,
        type=str
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: the data file are JSON files
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.dev_file is not None:
        data_files["dev"] = args.dev_file
    # data_files["support"] = "dataset/structshot/support-conll-5shot/0.json"
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if args.label_map_path is not None:
        print("Loading label map from {}...".format(args.label_map_path))
        import json
        ori_label_token_map = json.load(open(args.label_map_path, 'r'))
    else:
        ori_label_token_map = {"I-PER":['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'], "I-ORG": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                           "I-LOC":['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'], "I-MISC":['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic']}

    # if args.label_schema == "BIO":
    #     new_ori_label_token_map = {}
    #     for key, value in ori_label_token_map.items():
    #         new_ori_label_token_map[key] = value
    #         new_ori_label_token_map["B-"+key[2:]] = value
    #     ori_label_token_map = new_ori_label_token_map

    label_list = list(ori_label_token_map.keys())
    label_list += 'O'

    label_to_id = {"O":0}
    for l in label_list:
        if l != "O":
            label_to_id[l] = len(label_to_id)
    # label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    print(ori_label_token_map)
    if True: #args.eval_label_schema == "BIO":
        import copy
        new_label_to_id = copy.deepcopy(label_to_id)
        for label, id in label_to_id.items():
            if label != "O" and "B-"+label[2:] not in label_to_id:
                new_label_to_id["B-"+label[2:]] = len(new_label_to_id)
        label_to_id = new_label_to_id
    id_to_label = {id:label for label,id in label_to_id.items()}
    print(label_to_id)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, do_lower_case=False)

    if args.model_name_or_path:
        if "xlnet" in args.model_name_or_path:
            model = XLNetLMHeadModel.from_pretrained(
                args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,)
        else:
            model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    if "roberta" in args.model_name_or_path:
        tokenizer = add_label_token_roberta(model, tokenizer, ori_label_token_map)
    elif "bert" in args.model_name_or_path:
        tokenizer = add_label_token_bert(model, tokenizer, ori_label_token_map)
    else:
        pass
    label_token_map = {item:item for item in ori_label_token_map}
    # label_token_map = ori_label_token_map
    print(ori_label_token_map)
    label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in label_token_map.items()}
    label_token_id_to_label = {idx:label for label,idx in label_token_to_id.items()}

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        
        target_tokens = []
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            input_ids = tokenized_inputs.input_ids[i]
            previous_word_idx = None
            label_ids = []
            target_token = []
            for input_idx, word_idx in zip(input_ids, word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.

                if word_idx is None:
                    target_token.append(-100)
                    label_ids.append(-100)
                # Set target token for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    if args.label_schema == "IO" and label[word_idx] != "O":
                        label[word_idx] = "I-"+label[word_idx][2:]

                    if label[word_idx] !='O':
                        target_token.append(label_token_to_id[label[word_idx]])
                    else:
                        target_token.append(input_idx)
                    # target_tokens.append()

                # Set target token for other tokens of each word.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                    if args.label_schema == "IO" and label[word_idx] != "O":
                        label[word_idx] = "I-"+label[word_idx][2:]

                    if label[word_idx] !='O':
                        # Set the same target token for each tokens.
                        target_token.append(label_token_to_id[label[word_idx]])

                        # Ignore the other words during training.
                        # target_token.append(-100)
                    else:
                        target_token.append(input_idx)
                previous_word_idx = word_idx
            target_tokens.append(target_token)
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = target_tokens
        tokenized_inputs['ori_labels'] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    # dev_dataset = processed_raw_datasets["dev"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForLMTokanClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    # model, optimizer, train_dataloader, eval_dataloader, dev_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, dev_dataloader
    # )

    # label_id_list = torch.tensor(list(label_token_to_id.values()), dtype=torch.long, device=device)
    label_id_list = torch.tensor([label_token_to_id[id_to_label[i]] for i in range(len(id_to_label)) if i != 0 and not id_to_label[i].startswith("B-")], dtype=torch.long, device=device)
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    print(label_id_list)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("./seqeval_metric.py")
    label_schema = args.label_schema


    def switch_to_BIO(labels):
        past_label = 'O'
        labels_BIO = []
        for label in labels:
            if label.startswith('I-') and (past_label=='O' or past_label[2:]!=label[2:]):
                labels_BIO.append('B-'+label[2:])
            else:
                labels_BIO.append(label)
            past_label = label
        return labels_BIO


    def get_labels(predictions, references, tokens, emissions=None, viterbi_decoder=None):

        use_crf = True if emissions is not None else False
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
            x_tokens = tokens.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()
            x_tokens = tokens.detach().cpu().clone().tolist()
            if use_crf:
                emissions = emissions.detach().cpu().clone().numpy()

        if use_crf:
            # The viterbi decoding algorithm

            out_label_ids = y_true
            preds = y_pred

            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            emissions_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]

            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != -100:
                        out_label_list[i].append(id_to_label[out_label_ids[i][j]])
                        emissions_list[i].append(emissions[i][j])
                        preds_list[i].append(label_token_id_to_label[preds[i][j]] if preds[i][j] in label_token_id_to_label.keys() else 'O')

            preds_list = [[] for _ in range(out_label_ids.shape[0])]
            for i in range(out_label_ids.shape[0]):
                sent_scores = torch.tensor(emissions_list[i])
                sent_len, n_label = sent_scores.shape
                sent_probs = torch.nn.functional.softmax(sent_scores, dim=1)
                start_probs = torch.zeros(sent_len) + 1e-6
                sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
                feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label + 1))
                vit_labels = viterbi_decoder.viterbi(feats)
                vit_labels = vit_labels.view(sent_len)
                vit_labels = vit_labels.detach().cpu().numpy()
                for label in vit_labels:
                    preds_list[i].append(id_to_label[label - 1])

            true_predictions = preds_list
            true_labels = out_label_list
            ori_tokens = [
                [tokenizer.convert_ids_to_tokens(t) for (p, l, t) in zip(pred, gold_label, token) if l != -100]
                for pred, gold_label, token in zip(y_pred, y_true, x_tokens)
            ]

        else:
            # Remove ignored index (special tokens)
            # Here we only use the first token of each word for evaluation.
            true_predictions = [
                [label_token_id_to_label[p] if p in label_token_id_to_label.keys() else 'O' for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]

            true_labels = [
                [id_to_label[l] for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]

            ori_tokens = [
                [tokenizer.convert_ids_to_tokens(t) for (p, l, t) in zip(pred, gold_label, token) if l != -100]
                for pred, gold_label, token in zip(y_pred, y_true, x_tokens)
            ]

        # Turn the predictions into required label schema.
        if args.label_schema == "IO" and args.eval_label_schema == 'BIO':
            true_predictions = list(map(switch_to_BIO, true_predictions))
        if args.eval_label_schema == 'IO':
            true_labels = [['I-{}'.format(l[2:]) if l !='O' else 'O' for l in label] for label in true_labels]

        return true_predictions, true_labels, ori_tokens


    def compute_metrics():
        results = metric.compute()
        # print(results)
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"]
            }


    def evaluate(best_metric, load=False, use_crf=False):

        if load:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
            # unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

        if use_crf:
            abstract_transitions = get_abstract_transitions(args.crf_raw_path, "train")
            viterbi_decoder = ViterbiDecoder(len(label_list) + 1, abstract_transitions, 0.05)

        model.eval()
        start = time.time()
        token_list = []
        y_true = []
        y_pred = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                ner_label = batch.pop('ori_labels', 'not found ner_labels')
                outputs = model(**batch, output_hidden_states=True)

            predictions = outputs.logits.argmax(dim=-1)

            if use_crf:
                probs = torch.softmax(outputs.logits, -1)
                emissions = probs[:,:,label_id_list]
                O_emissions = probs[:,:,:label_id_list.min().data].max(-1)[0].unsqueeze(-1)
                emissions = torch.cat([O_emissions, emissions], dim=-1)

            labels = ner_label
            token_labels = batch.pop("input_ids")
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                token_labels = accelerator.pad_across_processes(token_labels, dim=1, pad_index=-100)
                if use_crf:
                    emissions = accelerator.pad_across_processes(emissions, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            token_labels_gathered = accelerator.gather(token_labels)
            if use_crf:
                emissions_gathered = accelerator.gather(emissions)

            if use_crf:
                preds, refs, tokens = get_labels(predictions_gathered, labels_gathered, token_labels_gathered, emissions_gathered, viterbi_decoder)
            else:
                preds, refs, tokens = get_labels(predictions_gathered, labels_gathered, token_labels_gathered)

            token_list.extend(tokens)
            y_true.extend(refs)
            y_pred.extend(preds)

            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids


        # eval_metric = metric.compute()
        eval_metric = compute_metrics()

        print("Decoding time: {}s".format(time.time() - start))
        # accelerator.print(f"epoch {epoch}:", eval_metric)
        for key in eval_metric.keys():
            if "f1" in key and "overall" not in key:
                label = key[:-3]
                print("{}: {}, {}: {}, {}: {}, {}: {}".format(label + "_precision", eval_metric[label + "_precision"],
                                                              label + "_recall", eval_metric[label + "_recall"],
                                                              label + "_f1", eval_metric[label + "_f1"],
                                                              label + "_number", eval_metric[label + "_number"]))
        label = "overall"
        print("{}: {}, {}: {}, {}: {}, {}: {}".format(label + "_precision", eval_metric[label + "_precision"],
                                                      label + "_recall", eval_metric[label + "_recall"],
                                                      label + "_f1", eval_metric[label + "_f1"],
                                                      label + "_accuracy", eval_metric[label + "_accuracy"]))

        if best_metric == -1 or best_metric["overall_f1"] < eval_metric["overall_f1"] and not load:
            best_metric = eval_metric

            if args.output_dir is not None:
                # print(f"Save model to {args.output_dir}.")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(args.output_dir)

                with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
                    for i in range(len(y_true)):
                        for j in range(len(y_true[i])):
                            f.write(f"{token_list[i][j]} {y_true[i][j]} {y_pred[i][j]}\n")
                        f.write("\n")
        return best_metric

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_metric = -1
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # Test each epoch and save the model of the best epoch.
        # if epoch>=0:
        #     best_metric = evaluate(best_metric)

        # Use the result of the last epoch
        if epoch == args.num_train_epochs - 1:
            best_metric = evaluate(best_metric)

            print("Finish training, best metric: ")
            print(best_metric)

            if args.do_crf:
                print("Decoding with CRF: ")
                evaluate(best_metric=-1,load=False,use_crf=True)

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)



def nn_decode(reps, support_reps, support_tags):
    """
    NNShot: neariest neighbor decoder for few-shot NER
    """
    batch_size, sent_len, ndim = reps.shape
    scores = _euclidean_metric(reps.view(-1, ndim), support_reps, True)
    # tags = support_tags[torch.argmax(scores, 1)]
    emissions = get_nn_emissions(scores, support_tags)
    tags = torch.argmax(emissions, 1)
    return tags.view(batch_size, sent_len), emissions.view(batch_size, sent_len, -1)

def get_nn_emissions(scores, tags):
    """
    Obtain emission scores from NNShot
    """
    n, m = scores.shape
    n_tags = torch.max(tags) + 1
    emissions = -100000. * torch.ones(n, n_tags).to(scores.device)
    for t in range(n_tags):
        mask = (tags == t).float().view(1, -1)
        masked = scores * mask
        masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
        emissions[:, t] = torch.max(masked, dim=1)[0]
    return emissions

def _euclidean_metric(a, b, normalize=False):
    if normalize:
        a = torch.nn.functional.normalize(a)
        b = torch.nn.functional.normalize(b)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def get_abstract_transitions(data_dir, data_fname):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    examples = NER().read_examples_from_file(data_dir, data_fname)
    tag_lists = [example.labels for example in examples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans


class DataCollatorForLMTokanClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in ori_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in ori_labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


def add_label_token_bert(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.bert.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.bert.embeddings.word_embeddings.weight.data[indexes[0]]

            # Calculate mean vector if there are multiple label words.
            for i in indexes[1:]:
                embed += model.bert.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.bert.embeddings.word_embeddings.weight.data[index] = embed

    return tokenizer


def add_label_token_roberta(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.roberta.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.roberta.embeddings.word_embeddings.weight.data[indexes[0]]

            # Calculate mean vector if there are multiple label words.
            for i in indexes[1:]:
                embed += model.roberta.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.roberta.embeddings.word_embeddings.weight.data[index] = embed

    return tokenizer


def set_seed(seed=4):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)



if __name__ == "__main__":
    # set_seed()
    main()