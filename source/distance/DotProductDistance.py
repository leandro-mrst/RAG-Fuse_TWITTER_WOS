import torch
from pytorch_metric_learning.distances import BaseDistance


class DotProductDistance(BaseDistance):

    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def check_shapes(self, query_emb, ref_emb):
        pass

    def compute_mat(self, text_rpr, label_rpr):
        return 20 * torch.einsum("ab,cb->ac", text_rpr, label_rpr)

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError
