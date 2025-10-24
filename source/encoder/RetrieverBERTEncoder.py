import torch
from pytorch_lightning import LightningModule
from transformers import BertModel


class RetrieverBERTEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, output_hidden_states, pooling):
        super(RetrieverBERTEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states

        )
        self.pooling = pooling

    def forward(self, feature):
        attention_mask = torch.where(feature > 0, 1, 0)
        encoder_outputs = self.encoder(feature, attention_mask)
        return self.pooling(
            encoder_outputs,
            attention_mask
        )

