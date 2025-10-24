import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Retriever Predict Dataset.
    """

    def __init__(self, samples, text_features_source, tokenizer, text_max_length):
        super(TextDataset, self).__init__()
        self.texts = []
        self.text_features_source = text_features_source
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length

        for sample in tqdm(samples, desc="Reading Texts"):
            self.texts.append({
                "text_idx": sample["text_idx"],
                "text": self._get_features(sample),
            })

    def _get_features(self, sample):
        if self.text_features_source == "KWD":
            return " ".join([kwd[0] for kwd in sample["keywords"]])
        elif self.text_features_source == "TXT":
            return sample["text"]
        else:
            raise Exception("Source features must be TXT or KWD")

    def _encode(self, sample):
        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"], max_length=self.text_max_length, padding="max_length", truncation=True
                )),
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._encode(
            self.texts[idx]
        )
