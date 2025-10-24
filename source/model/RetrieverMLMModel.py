import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup

from source.metric.RetrieverMetric import RetrieverMetric


class RetrieverModel(LightningModule):
    """Encodes the text and label into a same space of embeddings."""

    def __init__(self, hparams):
        super(RetrieverModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        self.num_augmented_tokens = 32

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = RetrieverMetric(hparams.metric)

    def forward(self, text, label):
        # expanding the label with learned augmented tokens sparse_rpr, dense_rpr
        sparse_label_rpr, _ = self.encoder(label)
        _, tokens_ids = torch.sort(sparse_label_rpr, dim=-1, descending=True)
        augmented_label = torch.cat((label, tokens_ids[:, :self.num_augmented_tokens]), dim=-1)
        _, text_rpr = self.encoder(text)
        _, label_rpr = self.encoder(augmented_label)
        return text_rpr, label_rpr

    def training_step(self, batch, batch_idx):
        text_rpr, label_rpr = self(batch["text"], batch["label"])
        train_loss = self.loss(batch["text_idx"], text_rpr, batch["label_idx"], label_rpr)
        self.log('train_LOSS', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        text_rpr, label_rpr = self(batch["text"], batch["label"])
        self.mrr.update(batch["text_idx"], text_rpr, batch["label_idx"], label_rpr)

    def on_validation_epoch_end(self):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            _, text_rpr = self.encoder(batch["text"])
            return {
                "text_idx": batch["text_idx"],
                "text_rpr": text_rpr,
                "modality": "text"
            }
        else:
            sparse_label_rpr, _ = self.encoder(batch["label"])
            _, tokens_ids = torch.sort(sparse_label_rpr, dim=-1, descending=True)
            augmented_label = torch.cat((batch["label"], tokens_ids[:, :self.num_augmented_tokens]), dim=-1)
            _, label_rpr = self.encoder(augmented_label)
            return {
                "label_idx": batch["label_idx"],
                "label_rpr": label_rpr,
                "modality": "label"
            }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
    #                                   eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
    #
    #     # schedulers
    #     step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)
    #
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
    #                                                   base_lr=self.hparams.base_lr,
    #                                                   max_lr=self.hparams.max_lr, step_size_up=step_size_up,
    #                                                   cycle_momentum=False)
    #
    #     return (
    #         {"optimizer": optimizer,
    #          "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
    #     )
