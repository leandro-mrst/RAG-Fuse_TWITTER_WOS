import logging
import pickle
from pathlib import Path

import nmslib
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

from source.helper.Helper import Helper


class RetrieverEvalHelper(Helper):
    def __init__(self, params):
        super(RetrieverEvalHelper, self).__init__()
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        self.samples_df = pd.DataFrame(self._load_samples())

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
            metrics.append(f"{metric}@{self.params.data.num_relevant_labels}")

        return metrics

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)

    def _get_split_texts_ids(self, fold_idx, split):
        split_sample_ids = self._load_split_ids(fold_idx, split)
        samples_df = self.samples_df[self.samples_df["idx"].isin(split_sample_ids)]
        return samples_df["text_idx"].to_list()

    def _get_split_labels_ids(self, fold_idx, split):
        labels_ids = set()
        split_sample_ids = self._load_split_ids(fold_idx, split)
        samples_df = self.samples_df[self.samples_df["idx"].isin(split_sample_ids)]
        for _, r in samples_df.iterrows():
            labels_ids.update(r["labels_ids"])
        return list(labels_ids)

    def _load_predictions(self, fold_idx, split):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold_idx}/").glob("*.prd")
        )

        split_texts_ids = self._get_split_texts_ids(fold_idx, split)
        split_labels_ids = self._get_split_labels_ids(fold_idx, split)

        text_predictions = []
        label_predictions = []

        for path in tqdm(predictions_paths, desc="Loading predictions"):
            for prediction in torch.load(path):

                if prediction["modality"] == "text" and prediction["text_idx"] in split_texts_ids:
                    text_predictions.append({
                        "text_idx": prediction["text_idx"],
                        "text_rpr": prediction["text_rpr"]
                    })

                elif prediction["modality"] == "label" and prediction["label_idx"] in split_labels_ids:
                    label_predictions.append({
                        "label_idx": prediction["label_idx"],
                        "label_rpr": prediction["label_rpr"]
                    })
        logging.info(f"\n{split}: added {len(text_predictions)} texts")
        logging.info(f"{split}: added {len(label_predictions)} labels\n")

        return text_predictions, label_predictions

    def init_index(self, label_predictions, cls):
        added = 0
        index = nmslib.init(method='hnsw', space='l2')
        for prediction in tqdm(label_predictions, desc="Adding data to index"):
            if cls in self.labels_cls[prediction["label_idx"]]:
                index.addDataPoint(id=prediction["label_idx"], data=prediction["label_rpr"])
                added += 1

        index.createIndex(
            index_params=OmegaConf.to_container(self.params.eval.index),
            print_progress=False
        )
        logging.info(f"Added {added} labels.")
        return index

    # def retrieve(self, index, text_predictions, cls, num_nearest_neighbors):
    #     # retrieve
    #     ranking = {}
    #     index.setQueryTimeParams({'efSearch': 2048})
    #     for prediction in tqdm(text_predictions, desc="Searching"):
    #         text_idx = prediction["text_idx"]
    #         if cls in self.texts_cls[text_idx]:
    #             retrieved_ids, distances = index.knnQuery(prediction["text_rpr"], k=num_nearest_neighbors)
    #             for label_idx, distance in zip(retrieved_ids, distances):
    #                 if f"text_{text_idx}" not in ranking:
    #                     ranking[f"text_{text_idx}"] = {}
    #                 ranking[f"text_{text_idx}"][f"label_{label_idx}"] = 1.0 / (distance + 1e-9)
    #
    #     return ranking

    def retrieve(self, index, text_predictions, cls, num_labels):
        # retrieve
        searched = 0
        ranking = {}
        index.setQueryTimeParams({'efSearch': 2048})
        for prediction in tqdm(text_predictions, desc="Searching"):
            text_idx = prediction["text_idx"]
            if cls in self.texts_cls[text_idx]:
                retrieved_ids, distances = index.knnQuery(prediction["text_rpr"], k=num_labels)
                for label_idx, distance in zip(retrieved_ids, distances):
                    if f"text_{text_idx}" not in ranking:
                        ranking[f"text_{text_idx}"] = {}
                    score = 1.0 / (distance + 1e-9)
                    if f"label_{label_idx}" in ranking[f"text_{text_idx}"]:
                        if score > ranking[f"text_{text_idx}"][f"label_{label_idx}"]:
                            ranking[f"text_{text_idx}"][f"label_{label_idx}"] = score
                    else:
                        ranking[f"text_{text_idx}"][f"label_{label_idx}"] = score
                searched += 1
        logging.info(f"Searched {searched} texts.")
        return ranking

    def _get_ranking(self, text_predictions, label_predictions, cls, num_labels):
        # index data
        index = self.init_index(label_predictions, cls)
        # retrieve
        return self.retrieve(index, text_predictions, cls, num_labels)

    def perform_eval(self):
        rankings = {}
        for fold_idx in self.params.data.folds:
            results = []
            rankings[fold_idx] = {}
            for split in ["test"]:
                rankings[fold_idx][split] = {}
                logging.info(
                    f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                    f"{OmegaConf.to_yaml(self.params)}\n")

                text_predictions, label_predictions = self._load_predictions(fold_idx, split)

                for cls in self.params.eval.label_cls:
                    logging.info(f"Evaluating {cls} ranking")
                    ranking = self._get_ranking(text_predictions, label_predictions, cls=cls,
                                                num_labels=self.params.eval.num_nearest_neighbors)
                    result = evaluate(
                        Qrels(
                            {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                        ),
                        Run(ranking),
                        self.metrics
                    )
                    result = {k: round(v, 3) for k, v in result.items()}
                    result["fold_idx"] = fold_idx
                    result["split"] = split
                    result["cls"] = cls

                    results.append(result)
                    rankings[fold_idx][split][cls] = ranking

            self.checkpoint_ranking(rankings[fold_idx], fold_idx)
            self._checkpoint_fold_results(results, fold_idx)


