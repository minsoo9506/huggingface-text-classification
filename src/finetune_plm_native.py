import argparse
import random
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizerFast,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from data_load.dataset import TextClassificationDataset, TextClssificationCollator
from lit_model.lit_bert import LitBert
from utils.utils import read_text

# import torch_optimizer as custom_optim


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--train_fn", required=True)
    p.add_argument("--model_fn", type=str, default="../save_model/model.pth")

    p.add_argument("--pretrained_model_name", type=str, default="beomi/kcbert-base")
    p.add_argument("--use_albert", action="store_true")

    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=1)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)

    p.add_argument("--use_radam", action="store_true")
    p.add_argument("--valid_ratio", type=float, default=0.2)

    p.add_argument("--max_length", type=int, default=100)

    config = p.parse_args()

    return config


def get_loader(
    fn: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    valid_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """make train, valid dataloader

    Parameters
    ----------
    fn : str
        _description_
    tokenizer : PreTrainedTokenizer
        _description_
    max_length : int
        max length of text
    valid_ratio : float, optional
        _description_, by default 0.2

    Returns
    -------
    Tuple[DataLoader, DataLoader, Dict]
        _description_
    """
    labels, texts = read_text(fn)

    # label을 int로 바꾸기
    unique_labels = list(set(labels))
    label_to_int = {}
    int_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
        int_to_label[i] = label
    labels = list(map(label_to_int.get, labels))

    # shuffle -> split train, valid
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    # dataloader
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClssificationCollator(tokenizer, max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClssificationCollator(tokenizer, max_length),
    )

    return train_loader, valid_loader, int_to_label


def get_optimizer(model: Any, config: argparse.Namespace):
    """return optimizer

    Parameters
    ----------
    model : Any
        _description_
    config : argparse.Namespace
        _description_
    """

    if not config.use_radam:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon
        )

    # else:
    #     optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)

    return optimizer


def main(config: argparse.Namespace):
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

    train_loader, valid_loader, int_to_label = get_loader(
        config.train_fn, tokenizer, config.max_length, config.valid_ratio
    )

    print("|train| = ", len(train_loader) * config.batch_size)
    print("|valid| = ", len(valid_loader) * config.batch_size)

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    print("# total iter = ", n_total_iterations)
    print("# warmup iter = ", n_warmup_steps)

    model_loader = (
        AlbertForSequenceClassification
        if config.use_albert
        else BertForSequenceClassification
    )
    model = model_loader.from_pretrained(
        config.pretrained_model_name, num_labels=len(int_to_label)
    )
    optimizer = get_optimizer(model, config)
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, n_warmup_steps, n_total_iterations
    )

    # if config.gpu_id >= 0:
    #     model.cuda(config.gpu_id)
    #     crit.cuda(config.gpu_id)

    lit_model = LitBert(model, crit, optimizer, scheduler)

    logger = TensorBoardLogger(save_dir="../log", name="bert_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="../save_model",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16,
    )
    trainer.fit(lit_model, train_loader, valid_loader)

    torch.save(
        {
            "bert": model.state_dict(),
            "config": config,
            "vocab": None,
            "classes": int_to_label,
            "tokenizer": tokenizer,
        },
        config.model_fn,
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
