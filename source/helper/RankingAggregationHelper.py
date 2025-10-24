import logging
import pickle
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from omegaconf import OmegaConf
from ranx import evaluate, Qrels, Run
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from source.helper.Helper import Helper


@nb.njit()
def in1d(a, b):
    """
    asssume a and b are unique
    """
    arr = np.concatenate((a, b))
    order = arr.argsort(kind='mergesort')
    sarr = arr[order]

    bool_arr = (sarr[1:] == sarr[:-1])
    flag = np.concatenate((bool_arr, np.asarray([False])))
    ret = np.empty(arr.shape, np.bool_)

    ret[order] = flag

    return ret[:len(a)]


@nb.njit(parallel=True)
def _topk_nb(data, indices, indptr, k, pad_ind, pad_val):
    """Get top-k indices and values for a sparse (csr) matrix
    * Parallel version: uses numba
    Arguments:
    ---------
    data: np.ndarray
        data / vals of csr array
    indices: np.ndarray
        indices of csr array
    indptr: np.ndarray
        indptr of csr array
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    nr = len(indptr) - 1
    ind = np.full((nr, k), fill_value=pad_ind, dtype=indices.dtype)
    val = np.full((nr, k), fill_value=pad_val, dtype=data.dtype)

    for i in nb.prange(nr):
        s, e = indptr[i], indptr[i + 1]
        num_el = min(k, e - s)
        temp = np.argsort(data[s: e])[::-1][:num_el]
        ind[i, :num_el] = indices[s: e][temp]
        val[i, :num_el] = data[s: e][temp]
    return ind, val


def topk(X, k, pad_ind, pad_val, return_values=False,
         dtype='float32', use_cython=False):
    """Get top-k indices and values for a sparse (csr) matrix
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    return_values: boolean, optional, default=False
        Return topk values or not
    dtype: str, optional, default='float32'
        datatype of values
    use_cython: bool, optional, default=False
        use cython to compute topk
        may be helpful when numba isn't working on some machine

    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    ind, val = _topk_nb(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    if return_values:
        return ind, val.astype(dtype)
    else:
        return ind


def compatible_shapes(x, y):
    """
    See if both matrices have same shape

    Works fine for the following combinations:
    * both are sparse
    * both are dense

    Will only compare rows when:
    * one is sparse/dense and other is dict
    * one is sparse and other is dense

    ** User must ensure that predictions are of correct shape when a
    np.ndarray is passed with all predictions.
    """
    # both are either sparse or dense
    if (sp.issparse(x) and sp.issparse(y)) \
            or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return x.shape == y.shape

    # compare #rows if one is sparse and other is dict or np.ndarray
    if not (isinstance(x, dict) or isinstance(y, dict)):
        return x.shape[0] == y.shape[0]
    else:
        if isinstance(x, dict):
            return len(x['indices']) == len(x['scores']) == y.shape[0]
        else:
            return len(y['indices']) == len(y['scores']) == x.shape[0]


def format(*args, decimal_points='%0.2f'):
    out = []
    for vals in args:
        out.append(
            ','.join(list(map(lambda x: decimal_points % (x * 100), vals))))
    return '\n'.join(out)


def _broad_cast(mat, like):
    if isinstance(like, np.ndarray):
        return np.asarray(mat)
    elif sp.issparse(mat):
        return mat
    else:
        raise NotImplementedError(
            "Unknown type; please pass csr_matrix, np.ndarray or dict.")


def _get_topk_sparse(X, pad_indx=0, k=5, use_cython=False):
    """
    Get top-k elements when X is a sparse matrix
    * Support for cython (use_cython=True) and numba (use_cython=False)
    """
    X = X.tocsr()
    X.sort_indices()
    pad_indx = X.shape[1]
    indices = topk(
        X, k, pad_indx, 0, return_values=False, use_cython=use_cython)
    return indices


def _get_topk_array(X, k=5, sorted=False):
    """
    Get top-k elements when X is an array
    X can be an array of:
        indices: indices of top predictions (must be sorted)
        values: scores for all labels (like in one-vs-all)
    """
    # indices are given
    assert X.shape[1] >= k, "Number of elements in X is < {}".format(k)
    if np.issubdtype(X.dtype, np.integer):
        assert sorted, "sorted must be true with indices"
        indices = X[:, :k] if X.shape[1] > k else X
    # values are given
    elif np.issubdtype(X.dtype, np.floating):
        _indices = np.argpartition(X, -k)[:, -k:]
        _scores = np.take_along_axis(
            X, _indices, axis=-1
        )
        indices = np.argsort(-_scores, axis=-1)
        indices = np.take_along_axis(_indices, indices, axis=1)
    return indices


def _get_topk_dict(X, k=5, sorted=False):
    """
    Get top-k elements when X is an dict of indices and scores
    X['scores'][i, j] will contain score of
        ith instance and X['indices'][i, j]th label
    """
    indices = X['indices']
    scores = X['scores']
    assert compatible_shapes(indices, scores), \
        "Dimension mis-match: expected array of shape {} found {}".format(
            indices.shape, scores.shape)
    assert scores.shape[1] >= k, "Number of elements in X is < {}".format(
        k)
    # assumes indices are already sorted by the user
    if sorted:
        return indices[:, :k] if indices.shape[1] > k else indices

    # get top-k entried without sorting them
    if scores.shape[1] > k:
        _indices = np.argpartition(scores, -k)[:, -k:]
        _scores = np.take_along_axis(
            scores, _indices, axis=-1
        )
        # sort top-k entries
        __indices = np.argsort(-_scores, axis=-1)
        _indices = np.take_along_axis(_indices, __indices, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    else:
        _indices = np.argsort(-scores, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    return indices


def _get_topk(X, pad_indx=0, k=5, sorted=False, use_cython=False):
    """
    Get top-k indices (row-wise); Support for
    * csr_matirx
    * 2 np.ndarray with indices and values
    * np.ndarray with indices or values
    """
    if sp.issparse(X):
        indices = _get_topk_sparse(
            X=X,
            pad_indx=pad_indx,
            k=k,
            use_cython=use_cython)
    elif isinstance(X, np.ndarray):
        indices = _get_topk_array(
            X=X,
            k=k,
            sorted=sorted)
    elif isinstance(X, dict):
        indices = _get_topk_dict(
            X=X,
            k=k,
            sorted=sorted)
    else:
        raise NotImplementedError(
            "Unknown type; please pass csr_matrix, np.ndarray or dict.")
    return indices


def compute_inv_propesity(labels, A, B):
    """
    Computes inverse propernsity as proposed in Jain et al. 16.

    Arguments:
    ---------
    labels: csr_matrix
        label matrix (typically ground truth for train data)
    A: float
        typical values:
        * 0.5: Wikipedia
        * 0.6: Amazon
        * 0.55: otherwise
    B: float
        typical values:
        * 0.4: Wikipedia
        * 2.6: Amazon
        * 1.5: otherwise

    Returns:
    -------
    np.ndarray: propensity scores for each label
    """
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances) - 1) * np.power(B + 1, A)
    wts = 1.0 + C * np.power(freqs + B, -A)
    return np.ravel(wts)


def _setup_metric(X, true_labels, inv_psp=None,
                  k=5, sorted=False, use_cython=False):
    assert compatible_shapes(X, true_labels), \
        "ground truth and prediction matrices must have same shape."
    num_instances, num_labels = true_labels.shape
    indices = _get_topk(X, num_labels, k, sorted, use_cython)
    ps_indices = None
    if inv_psp is not None:
        _mat = sp.spdiags(inv_psp, diags=0,
                          m=num_labels, n=num_labels)
        _psp_wtd = _broad_cast(_mat.dot(true_labels.T).T, true_labels)
        ps_indices = _get_topk(_psp_wtd, num_labels, k, False, use_cython)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])

    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels + 1), dtype=true_labels.dtype)

    # scipy won't respect the dtype of indices
    # may fail otherwise on really large datasets
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        nr, nc = indices.shape
        rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
        eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels,
                                        indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def psprecision(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    """
    Compute propensity scored precision@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored precision till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system

    Returns:
    -------
    np.ndarray: propensity scored precision values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _precision(eval_flags, k) / _precision(ps_eval_flags, k)


def _precision(eval_flags, k=5):
    deno = 1 / (np.arange(k) + 1)
    precision = np.mean(
        np.multiply(np.cumsum(eval_flags, axis=-1), deno),
        axis=0)
    return np.ravel(precision)


def psndcg(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    """
    Compute propensity scored nDCG@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored nDCG till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: propensity scored nDCG values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1 / np.log2(np.arange(1, _max_pos + 1) + 1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k) / _ndcg(ps_eval_flags, n, k)


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(
        eval_flags, 1 / np.log2(np.arange(k) + 2)),
        axis=-1)
    ndcg = np.zeros((1, k), dtype=np.float32)
    for _k in range(k):
        _cumsum += 1 / np.log2(_k + 1 + 1)
        ndcg[0, _k] = np.mean(
            np.multiply(_dcg[:, _k].reshape(-1, 1), 1 / np.minimum(n, _cumsum))
        )
    return np.ravel(ndcg)


class RankingAggregationHelper(Helper):

    def __init__(self, params):
        super(RankingAggregationHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.relevance_map = self._load_relevance_map()
        self.label_cls = self._load_labels_cls()
        self.text_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Aggregating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")

            labels_ids = []
            for sample in self.samples:
                labels_ids.append(sample['labels_ids'])

            mlb = MultiLabelBinarizer(sparse_output=True)
            labels = mlb.fit_transform(labels_ids)
            inv_propensity = compute_inv_propesity(labels, self.params.data.propensity.A,
                                                   self.params.data.propensity.B)

            ranking = self._load_ranking(fold_idx=fold_idx)
            aggregated_ranking = self._aggregate_ranking(
                tail_ranking=ranking["tail"],
                head_ranking=ranking["head"])

            result = self._eval_combined_ranking(
                aggregated_ranking,
                self.relevance_map,
                inv_propensity.shape[0],
                inv_propensity,
                self.params.eval.thresholds
            )
            result["fold_idx"] = fold_idx

            # save ranking
            self._checkpoint_ranking(aggregated_ranking, fold_idx)

            # save result
            self._checkpoint_result(result, fold_idx)

    def _checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}Aggregated_{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}Aggregated_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _checkpoint_result(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}Aggregated_{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame([result]).to_csv(
            f"{result_dir}Aggregated_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)

    # def _load_ranking(self, fold_idx):
    #     with open(
    #             f"{self.params.ranking.dir}Fused_"
    #             f"{self.params.model.name}_{self.params.data.name}/Fused_"
    #             f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
    #             "rb") as ranking_file:
    #         return pickle.load(ranking_file)

    def _load_ranking(self, fold_idx):
        with open(
                f"{self.params.ranking.dir}Fused_"
                f"{self.params.model.name}_{self.params.data.name}/Fused_"
                f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                "rb") as ranking_file:
            return pickle.load(ranking_file)

    def _aggregate_ranking(self, tail_ranking, head_ranking):
        aggregated_ranking = {}
        texts_ids = set(tail_ranking.keys()).union(set(head_ranking.keys()))
        for text_idx in texts_ids:
            aggregated_ranking[text_idx] = {}
            if text_idx in tail_ranking:
                aggregated_ranking[text_idx].update(tail_ranking[text_idx])
            if text_idx in head_ranking:
                aggregated_ranking[text_idx].update(head_ranking[text_idx])
        return aggregated_ranking

    def _eval_combined_ranking(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        # Get propensity scored metrics
        ps_results = self._compute_ps_metrics(ranking, relevance_map, num_labels, inv_propesities, thresholds)

        # Get traditional ranking metrics
        tr_results = self._compute_tr_metrics(ranking)

        # Combine results into a single dictionary
        results = {}
        results.update(ps_results)
        results.update(tr_results)

        return results

    def _compute_ps_metrics(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        text_ids_map = {k: v for v, k in enumerate(ranking.keys())}
        p_rows, p_cols, p_scores = [], [], []
        t_rows, t_cols, t_scores = [], [], []
        for text_idx, labels_scores in ranking.items():
            for label_idx, score in labels_scores.items():
                if int(label_idx.split("_")[-1]) >= 0:
                    p_rows.append(text_ids_map[text_idx])
                    p_cols.append(int(label_idx.split("_")[-1]))
                    p_scores.append(score)

            for label_idx in relevance_map[text_idx]:
                t_rows.append(text_ids_map[text_idx])
                t_cols.append(int(label_idx.split("_")[-1]))
                t_scores.append(1.0)

        pred = csr_matrix((p_scores, (p_rows, p_cols)), shape=(len(text_ids_map), num_labels))
        true = csr_matrix((t_scores, (t_rows, t_cols)), shape=(len(text_ids_map), num_labels))

        return self.__compute_ps_metrics(pred, true, inv_propesities, thresholds)

    def __compute_ps_metrics(self, pred, true, inv_propesities, thresholds):
        psprecisions = psprecision(pred, true, inv_propesities, k=thresholds[-1])
        psndcgs = psndcg(pred, true, inv_propesities, k=thresholds[-1])
        results = {}
        for k in thresholds:
            results[f"psnDCG@{k}"] = round(100 * psndcgs[k - 1], 1)

        for k in thresholds:
            results[f"psprecision@{k}"] = round(100 * psprecisions[k - 1], 1)

        return results

    def _compute_tr_metrics(self, ranking):
        result = evaluate(
            Qrels(
                {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
            ),
            Run(ranking),
            self.metrics
        )
        return {k: round(100 * v, 1) for k, v in result.items()}
