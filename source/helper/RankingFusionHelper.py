import logging
import pickle
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from ranx import fuse, Run, Qrels, evaluate

from source.helper.Helper import Helper


class RankingFusionHelper(Helper):

    def __init__(self, params):
        super(RankingFusionHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.relevance_map = self._load_relevance_map()
        self.label_cls = self._load_labels_cls()
        self.text_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        results = []
        rankings = {}

        for fold_idx in self.params.data.folds:
            logging.info(
                f"Fusing {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")
            rankings[fold_idx] = {}
            for cls in ["tail", "head"]:
                r1 = self._load_ranking(model_name="BM25", fold_idx=fold_idx)["test"][cls]
                r2 = self._load_ranking(model_name=self.params.model.name, fold_idx=fold_idx)["test"][cls]

                fused_ranking = fuse(
                    runs=[Run(r1), Run(r2)],  # A list of Run instances to fuse
                    norm=self.params.fusion.norm,  # The normalization strategy to apply before fusion
                    method=self.params.fusion.method  # The fusion algorithm to use
                ).to_dict()

                result = evaluate(
                    Qrels(
                        {key: value for key, value in self.relevance_map.items() if key in fused_ranking.keys()}
                    ),
                    Run(fused_ranking),
                    self.metrics
                )
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold_idx"] = fold_idx
                result["split"] = "test"
                result["cls"] = cls
                results.append(result)
                rankings[fold_idx][cls] = fused_ranking

            self._checkpoint_ranking(rankings[fold_idx], fold_idx)
            self._checkpoint_fold_results(results, fold_idx)

    def _checkpoint_fold_results(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}Fused_{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame(result).to_csv(
            f"{result_dir}Fused_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)

    def _checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}Fused_{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}Fused_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _load_ranking(self, model_name, fold_idx):
        with open(
                f"{self.params.ranking.dir}{model_name}_{self.params.data.name}/{model_name}_{self.params.data.name}_{fold_idx}.rnk",
                "rb") as ranking_file:
            return pickle.load(ranking_file)
