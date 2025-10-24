import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LabelDataset(Dataset):
    """Retriever Predict Dataset.
    """

    def __init__(self,
                 samples,
                 labels_descriptions,
                 pseudo_labels,
                 label_enhancement,
                 tokenizer,
                 label_max_length
                 ):

        super(LabelDataset, self).__init__()

        self.labels = []
        self.tokenizer = tokenizer
        self.label_max_length = label_max_length
        self.label_enhancement = label_enhancement
        self.labels_descriptions = labels_descriptions
        self.pseudo_labels = pseudo_labels

        unique_labels = {}

        for sample in tqdm(samples, desc="Reading Labels"):
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                unique_labels[label_idx] = f"{label} " + self._get_label_features(
                    label_idx)  # labels_descriptions.get(label_idx, f"{label}")

        for label_idx, label in unique_labels.items():
            self.labels.append({
                "label_idx": label_idx,
                "label": label
            })

    def _get_label_features(self, label_idx):
        if self.label_enhancement == "PMI":
            return self._get_pseudo_labels(label_idx)
        elif self.label_enhancement == "LLM":
            return self.labels_descriptions.get(label_idx, "")
        elif self.label_enhancement == "NONE":
            return ""
        else:
            raise Exception("Source features must be in [PMI, LLM, NONE]")

    def _get_pseudo_labels(self, label_idx):
        if self.label_enhancement:
            if label_idx in self.pseudo_labels:
                terms_scores = sorted(self.pseudo_labels[label_idx], key=lambda x: x[1], reverse=True)
                return " ".join(x[0] for x in terms_scores)
        return ""

    def _encode(self, sample):
        return {
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.label_max_length, padding="max_length", truncation=True
                )
            )
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._encode(
            self.labels[idx]
        )
