import argparse
import random
from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (AlbertForSequenceClassification,
                          BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)

from data_load.dataset import (TextClassificationDataset,
                               TextClssificationCollator)
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

    p.add_argument("--batch_size_per_device", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=1)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)

    p.add_argument("--use_radam", action="store_true")
    p.add_argument("--valid_ratio", type=float, default=0.2)

    p.add_argument("--max_length", type=int, default=100)

    config = p.parse_args()

    return config


def get_datasets(fn: str, valid_ratio: float = 0.2) -> Tuple[Dataset, Dataset, Dict]:
    labels, texts = read_text(fn)

    unique_labels = list(set(labels))
    label_to_int = {}
    int_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
        int_to_label[i] = label

    labels = list(map(label_to_int.get, labels))

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
    valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])

    return train_dataset, valid_dataset, int_to_label


def main(config):
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    train_dataset, valid_dataset, int_to_label = get_datasets(
        config.train_fn, config.valid_ratio
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    model_loader = (
        AlbertForSequenceClassification
        if config.use_albert
        else BertForSequenceClassification
    )
    model = model_loader.from_pretrained(
        config.pretrained_model_name, num_labels=len(int_to_label)
    )

    training_args = TrainingArguments(
        output_dir="../hf_trainer_output",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextClssificationCollator(
            tokenizer, config.max_length, with_text=False
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

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
