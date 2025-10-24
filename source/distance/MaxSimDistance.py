import torch
from pytorch_metric_learning.distances import BaseDistance


class MaxSimDistance(BaseDistance):

    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def check_shapes(self, query_emb, ref_emb):
        pass

    def compute_mat(self, text_rpr, label_rpr):
        m = torch.einsum('b i j, c k j -> b c i k', text_rpr, label_rpr)
        m = torch.max(m, -1).values.sum(dim=-1)
        #return m
        return torch.nn.functional.normalize(m, p=2, dim=-1)

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError
