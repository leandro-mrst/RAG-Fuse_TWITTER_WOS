import torch
from torch import nn
from pytorch_metric_learning import miners, losses
from source.distance.DotProductDistance import DotProductDistance
from source.miner.RelevanceMiner import RelevanceMiner
from pytorch_metric_learning.utils import common_functions as c_f


class RetrieverLoss(nn.Module):

    def __init__(self, params):
        c_f.check_shapes = lambda x, y: None
        super(RetrieverLoss, self).__init__()
        self.miner = RelevanceMiner(params.miner)
        self.criterion = losses.NTXentLoss(temperature=params.criterion.temperature, distance=DotProductDistance())

    def forward(self, text_idx, text_rpr, label_idx, label_rpr):
        """
        Computes the NTXentLoss.
        """
        miner_outs = self.miner.mine(text_ids=text_idx, label_ids=label_idx)
        return self.criterion(text_rpr, None, miner_outs, label_rpr, None)
