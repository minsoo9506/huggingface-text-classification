import pytorch_lightning as pl
import torch


class LitBert(pl.LightningModule):
    def __init__(self, model, crit, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["input_ids"], train_batch["labels"]
        mask = train_batch["attention_mask"]

        y_hat = self.model(x, attention_mask=mask).logits
        loss = self.crit(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch["input_ids"], valid_batch["labels"]
        mask = valid_batch["attention_mask"]

        y_hat = self.model(x, attention_mask=mask).logits
        loss = self.crit(y_hat, y)
        acc = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        self.log_dict({"valid_loss": loss, "valid_acc": acc})
