import pickle

import torch
from pytorch_metric_learning.distances import BaseDistance


class RewardDistance(BaseDistance):

    def __init__(self, params):
        super().__init__(params, is_inverted=True)
        assert self.is_inverted
        with open(f"{params.data.dir}propesities.pkl", "rb") as propesities_file:
            self.propesities = pickle.load(propesities_file)

    def check_shapes(self, query_emb, ref_emb):
        pass

    def compute_mat(self, text_rpr, label_rpr):
        return 20 * torch.einsum("ab,cb->ac", text_rpr, label_rpr)

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError
