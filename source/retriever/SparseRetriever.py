import retriv


class SparseRetriever:
    def __init__(self, index_name, model):
        self.sr = retriv.SparseRetriever(
            index_name=index_name,
            model=model,
            min_df=1,
            tokenizer="word",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
            hyperparams={'b': 0.75, 'k1': 1.5}
        )

    def index(self, collection):
        self.sr.index(collection)
        return self.sr
