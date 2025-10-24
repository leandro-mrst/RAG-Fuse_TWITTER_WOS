from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class RetrieverPredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(RetrieverPredictionWriter, self).__init__(params.write_interval)
        self.params = params
        self.checkpoint_dir = f"{self.params.dir}fold_{self.params.fold_idx}/"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        predictions = []

        # print(f"\ntext_idx ({text_idx.shape}):\n {text_idx}\n")
        # print(f"\ntext_rpr ({text_rpr.shape}):\n {text_rpr}\n")
        # print(f"\nlabels_ids ({torch.flatten(labels_ids).shape}):\n {torch.flatten(labels_ids)}\n")
        # print(f"\nlabels_rpr ({labels_rpr.shape}):\n {labels_rpr}\n")

        if prediction["modality"] == "text":
            for text_idx, text_rpr in zip(
                     prediction["text_idx"].tolist(),
                    prediction["text_rpr"].tolist()):
                predictions.append({

                    "text_idx": text_idx,
                    "text_rpr": text_rpr,
                    "modality": "text"
                })

        elif prediction["modality"] == "label":
            for label_idx, label_rpr in zip(
                    prediction["label_idx"].tolist(),
                    prediction["label_rpr"].tolist()):
                predictions.append({
                    "label_idx": label_idx,
                    "label_rpr": label_rpr,
                    "modality": "label"
                })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        torch.save(
            predictions,
            f"{self.checkpoint_dir}{dataloader_idx}_{batch_idx}.prd"
        )
