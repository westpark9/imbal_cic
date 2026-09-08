"""Drop-in replacements for ``sklearn.ensemble._gb_losses`` (removed in
scikit-learn 1.3).  BoostPFN was written against scikit-learn 0.24, whose
private gradient-boosting loss classes exposed exactly the two methods the
boosting loop uses:

    loss(y, raw_predictions)                 -> scalar deviance (line search)
    loss.negative_gradient(y, raw, k=...)    -> residual for class k

``MultinomialDeviance`` is transcribed from scikit-learn 0.24.2
``sklearn/ensemble/_gb_losses.py`` (verified line-by-line against the PyPI sdist,
2026-09-04 review) so the CE boosting arithmetic is unchanged.  Inputs may be
numpy arrays or CPU torch tensors -- BoostPFN passes train labels as torch tensors.

``LeastSquaresError`` / ``BinomialDeviance`` are NOT usable replacements: under
0.24.2 BoostPFN's ``--loss MSE`` path could never execute (``__call__`` with an
(n, K) raw-prediction matrix raises ValueError and ``negative_gradient`` takes no
positional ``k``), and BinomialDeviance is K=1/sigmoid-based.  They raise
NotImplementedError here instead of inventing semantics that exist nowhere upstream.
"""
import os

import numpy as np
from scipy.special import logsumexp


def _np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


class MultinomialDeviance:
    """Multinomial deviance for K >= 2 classes, raw_predictions of shape (n, K)."""

    is_multi_class = True

    def __init__(self, n_classes):
        # 0.24.2: "MultinomialDeviance requires more than 2 classes."  BoostPFN builds
        # MultinomialDeviance(num_classes) unconditionally for --loss CE, so under its
        # pinned sklearn a BINARY dataset cannot run the CE path at all (the released
        # largedataset_boostpfn.py then logs a 'prediction failure').  The 2-class softmax
        # deviance is mathematically well defined, so an explicit opt-in is provided for
        # reproducing the paper's binary rows:  BOOSTPFN_ALLOW_BINARY_MULTINOMIAL=1
        if n_classes < 3 and os.environ.get("BOOSTPFN_ALLOW_BINARY_MULTINOMIAL", "0") != "1":
            raise ValueError(f"MultinomialDeviance requires more than 2 classes; got {n_classes} "
                             "(set BOOSTPFN_ALLOW_BINARY_MULTINOMIAL=1 to allow the 2-class softmax deviance)")
        self.K = int(n_classes)

    def __call__(self, y, raw_predictions, sample_weight=None):
        y = _np(y).astype(np.int64).ravel()
        raw = _np(raw_predictions).astype(np.float64)
        Y = np.zeros((y.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            Y[:, k] = y == k
        per_sample = -1 * (Y * raw).sum(axis=1) + logsumexp(raw, axis=1)
        return float(np.average(per_sample, weights=sample_weight))

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):
        y = _np(y).astype(np.float64).ravel()
        raw = _np(raw_predictions).astype(np.float64)
        return y - np.nan_to_num(np.exp(raw[:, k] - logsumexp(raw, axis=1)))


class LeastSquaresError:
    """0.24.2 squared error on a 1-D raw prediction.  BoostPFN's ``--loss MSE`` hands
    it an (n, K) matrix and a positional ``k``, which 0.24.2 rejects (ValueError /
    TypeError) -- so the path is not runnable upstream and is refused here too."""

    is_multi_class = False

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, y, raw_predictions, sample_weight=None):
        y = _np(y).astype(np.float64)
        raw = _np(raw_predictions).astype(np.float64)
        if raw.ndim != 1 and raw.shape[-1] != 1:
            raise NotImplementedError("LeastSquaresError on (n, K) raw predictions is not executable in "
                                      "scikit-learn 0.24.2 either; BoostPFN --loss MSE is unsupported")
        return float(np.average((y - raw.ravel()) ** 2, weights=sample_weight))

    def negative_gradient(self, y, raw_predictions, **kwargs):
        y = _np(y).astype(np.float64).ravel()
        raw = _np(raw_predictions).astype(np.float64)
        if raw.ndim != 1 and raw.shape[-1] != 1:
            raise NotImplementedError("LeastSquaresError.negative_gradient takes no class index k in 0.24.2")
        return y - raw.ravel()


class BinomialDeviance:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BinomialDeviance (K=1, sigmoid) is not used by BoostPFN's CE path; "
                                  "no faithful compat provided")


class ExponentialLoss:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ExponentialLoss is not used by BoostPFN's CE/MSE paths")
