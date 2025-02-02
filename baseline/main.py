import argparse
import logging
import os
import random
import json
from copy import deepcopy

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
from sklearn.metrics import precision_score, recall_score

import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    DebertaV2ForSequenceClassification,
    get_linear_schedule_with_warmup
)

from .dataset import (
    FactLinkingDataset,
    FactGenerationDataset,
    FactGenerationEvalDataset,
    SPECIAL_TOKENS,
    ADD_TOKENS_VALUES
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)

from .utils.metrics import (
    BLEU, METEOR, ROUGE
)

from .utils.model import (
    run_batch_generation_train,
    run_batch_generation_eval,
    run_batch_linking,
    softmax
)

from .utils.data import write_generation_preds, write_linking_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import wandb

from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)
rank_to_device = {0: 0, 1: 1, 2: 2, 3: 3}


def get_classes(task):
    if task.lower() == "generation":
        return FactGenerationDataset, FactGenerationEvalDataset, AutoModelForSeq2SeqLM, \
               run_batch_generation_train, run_batch_generation_eval
    elif task.lower() == "linking":
        return FactLinkingDataset, FactLinkingDataset, DebertaV2ForSequenceClassification, \
               run_batch_linking, run_batch_linking
    else:
        raise ValueError("args.task not in ['linking', generation'], got %s" % task)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    tb_writer = None
    # root = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/runs"
    root = "runs"
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join(root, args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank_to_device[args.local_rank]], output_device=rank_to_device[args.local_rank],
            find_unused_parameters=True
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility

    tr_loss = -1
    local_steps = -1

    scaler = GradScaler()

    for num_epoch in train_iterator:
        tr_loss = 0.0
        local_steps = 0
        report_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # all_preds = []
        # all_labels_batch = []

        for step, batch in enumerate(epoch_iterator):
            model.train()
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            #     with record_function("model_inference"):
            with autocast(dtype=torch.bfloat16):
                if args.task == "generation":
                    loss = run_batch_fn_train(args, model, batch, tokenizer)
                else:
                    loss, _, _, _ = run_batch_fn_train(args, model, batch, tokenizer)

            if torch.isnan(loss):
                print("Loss is NaN")

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # loss.backward()
                scaler.scale(loss).backward()

            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            # raise RuntimeError("Stop here") 

            tr_loss += loss.item()
            scaler.unscale_(optimizer)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/local_steps)

            report_loss += loss.item()
            if (step + 1) % 2000 == 0:

                results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
                for key, value in results.items():
                    wandb.log({"step": global_step,
                            f"eval_{key}": value})
                    
                wandb.log({"step": global_step,
                        "lr": scheduler.get_lr()[0],
                        # "eval_loss": tr_loss / local_steps,
                        "train_loss": report_loss / 2000})

                report_loss = 0.0

        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                wandb.log({"step": global_step,
                        f"eval_{key}": value})
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
            wandb.log({"step": global_step,
                    "lr": scheduler.get_lr()[0]})
                    # "eval_loss": tr_loss / local_steps})

            checkpoint_prefix = f"checkpoint-{args.run_name}"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, args.params_file.split("/")[-1]), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2)
            logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    eval_output_dir = None
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_dataset.args.eval_all_snippets:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    # linking
    data_infos = []
    all_preds = []
    all_labels = []

    # generation
    metrics = [BLEU(), METEOR(), ROUGE()]
    all_output_texts = []
    all_infos = {"context_ids": [], "turn_ids": [], "head_ids": []}
    cur_context_id = -1
    cur_turn_id = -1
    cur_head_id = -1
    cur_gen_beam = []
    cur_refs = []
    tr_loss = 0.0

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            if args.task == "generation":
                gen_text, truth_text, data_info = run_batch_fn(args, model, batch, tokenizer)
                for batch_id, gen_beam in enumerate(gen_text):
                    if data_info["context_ids"][batch_id] == cur_context_id and \
                            data_info["turn_ids"][batch_id] == cur_turn_id and \
                            data_info["head_ids"][batch_id] == cur_head_id:
                        pass
                    else:
                        if len(cur_refs) > 0:
                            for metric in metrics:
                                for gen_single in cur_gen_beam:
                                    if gen_single:
                                        if args.dataroot.split("/")[2] != "all_linked_head":
                                            metric.update((gen_single, cur_refs))

                        cur_refs = []
                        cur_gen_beam = deepcopy(gen_beam)
                        cur_context_id = data_info["context_ids"][batch_id]
                        cur_turn_id = data_info["turn_ids"][batch_id]
                        cur_head_id = data_info["head_ids"][batch_id]

                        all_output_texts.append(cur_gen_beam)
                        all_infos["context_ids"].append(cur_context_id)
                        all_infos["turn_ids"].append(cur_turn_id)
                        all_infos["head_ids"].append(cur_head_id)

                    cur_refs.append(truth_text[batch_id])
            else:
                loss, _, mc_logits, mc_labels = run_batch_fn(args, model, batch, tokenizer)
                data_infos.append(batch[-1])
                all_preds.append(mc_logits.detach().cpu().numpy())
                all_labels.append(mc_labels.detach().cpu().numpy())
                tr_loss += loss.item()

    if args.task == "generation":
        if len(cur_refs) > 0:
            for metric in metrics:
                for gen_single in cur_gen_beam:
                    if gen_single:
                        if args.dataroot.split("/")[2] != "all_linked_head":
                            metric.update((gen_single, cur_refs))

        if args.output_file:
            write_generation_preds(eval_dataset.dataset_walker, args.output_file, all_output_texts, all_infos)

        result = dict()
        if args.local_rank in [-1, 0]:
            if args.dataroot.split("/")[2] != "all_linked_head":
                output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results %s *****" % desc)
                    writer.write("***** Eval results %s *****\n" % desc)
                    for metric in metrics:
                        name = metric.name()
                        score = metric.compute()
                        result[name] = score
                        logger.info("  %s = %s", name, str(score))
                        writer.write("%s = %s\n" % (name, str(score)))
    else:
        all_labels = np.concatenate(all_labels)
        all_pred_ids = np.concatenate([np.argmax(logits, axis=1).reshape(-1) for logits in all_preds])
        all_pred_scores = np.concatenate([softmax(logits, axis=1) for logits in all_preds], axis=0)
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        precision = precision_score(all_labels, all_pred_ids)
        recall = recall_score(all_labels, all_pred_ids)
        f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))
        result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "loss": tr_loss / len(eval_dataloader)}
        if args.output_file:
            write_linking_preds(args.output_file, data_infos, all_pred_ids, all_pred_scores)

        if args.local_rank in [-1, 0]:
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results %s *****" % desc)
                writer.write("***** Eval results %s *****\n" % desc)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--max_tokens", type=int, default=-1,
                        help="Maximum length of input tokens, will override that value in config")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset")
    parser.add_argument("--eval_dataset", type=str, default="test",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead, "
                        "useful to take the outputs from the previous task in the pipe-lined evaluation")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    args.local_rank = -1
    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)
    
    args.params = params  # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.model_name_or_path = args.model_name_or_path
    dataset_args.window = args.dataroot.split("/")[3]

    wandb.init(project="Comfact",
               config=params)
    args.run_name = wandb.run.name

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(rank_to_device[args.local_rank])
        device = torch.device("cuda", rank_to_device[args.local_rank])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    dataset_class_train, dataset_class_eval, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args.task)

    if args.eval_only:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.add_tokens(ADD_TOKENS_VALUES)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    for name, param in model.named_parameters():
        if name.startswith('deberta.encoder.layer') and int(name.split('.')[3]) < params["num_frozen_layers"]:
            param.requires_grad = False
        elif name.startswith('deberta.embeddings'):
            param.requires_grad = False

    for i in range(params["num_frozen_layers"], 12):
        model.deberta.encoder.layer[i].output.dropout.drop_prob = args.dropout
        model.deberta.encoder.layer[i].attention.self.dropout.drop_prob = args.dropout
        model.deberta.encoder.layer[i].attention.output.dropout.drop_prob = args.dropout

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        if args.oversample is None:
            args.oversample = False
        train_dataset = dataset_class_train(dataset_args, tokenizer, split_type="train", oversample=args.oversample)
        eval_dataset = dataset_class_eval(dataset_args, tokenizer, split_type=args.eval_dataset)

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, args.params_file.split("/")[-1]), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class_eval(dataset_args, tokenizer, split_type=args.eval_dataset,
                                          labels=not args.no_labels, labels_file=args.labels_file)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "test")

    return result


if __name__ == "__main__":
    main()
