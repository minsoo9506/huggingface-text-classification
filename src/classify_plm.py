import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import (AlbertForSequenceClassification,
                          BertForSequenceClassification, BertTokenizerFast)


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", type=str, default="../save_model/model.pth")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--top_k", type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    lines = []

    for line in sys.stdin:
        if line.strip() != "":
            lines += [line.strip()]

    return lines


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location="cpu" if config.gpu_id < 0 else "cuda:%d" % config.gpu_id,
    )

    train_config = saved_data["config"]
    bert_base = saved_data["bert"]
    int_to_label = saved_data["classes"]

    lines = read_text()

    with torch.no_grad():
        tokenizer = BertTokenizerFast.from_pretrained(
            train_config.pretrained_model_name
        )
        model_loader = (
            AlbertForSequenceClassification
            if train_config.use_albert
            else BertForSequenceClassification
        )
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name, num_labels=len(int_to_label)
        )
        model.load_state_dict(bert_base)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx : idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch["input_ids"]
            x = x.to(device)
            mask = mini_batch["attention_mask"]
            mask = mask.to(device)

            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)

            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0)

        probs, indice = y_hats.cpu().topk(config.top_k)

        for i in range(len(lines)):
            sys.stdout.write(
                "%s\t%s\n"
                % (
                    " ".join(
                        [int_to_label[int(indice[i][j])] for j in range(config.top_k)]
                    ),
                    lines[i],
                )
            )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
