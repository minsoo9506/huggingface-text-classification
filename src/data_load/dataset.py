from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextClssificationCollator:
    """collator function"""

    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: int, with_text: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,  # 미니배치에서 가장 긴 길이에 맞춰서 padding
            truncation=True,  # max_length보다 긴 문장은 자르기
            return_tensors="pt",
            max_length=self.max_length,
        )

        return_value = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.with_text:
            return_value["text"] = texts

        return return_value


class TextClassificationDataset(Dataset):
    """torch Dataset

    Parameters
    ----------
    Dataset : _type_
        _description_
    """

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        return {"text": text, "label": label}
