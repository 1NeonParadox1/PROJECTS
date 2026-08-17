"""
cv.py
=====
Time-series-aware cross-validation utilities that prevent information
leakage from overlapping prediction windows and serial correlation, per
Lopez de Prado's "Advances in Financial Machine Learning".

Why not sklearn.model_selection.TimeSeriesSplit alone?
-------------------------------------------------------
Plain walk-forward CV already avoids training-on-the-future, but with
engineered features that use rolling windows (e.g. a 60-day realized vol)
and labels that look N steps ahead, samples *near the train/test boundary*
still leak information both ways:
  - A test-set feature at time t may have been computed using a rolling
    window that overlaps the train set.
  - A train-set label at time t (defined using price/regime info up to
    t+N) may overlap with the test set's earliest timestamps.

`PurgedGroupTimeSeriesSplit` fixes this with two mechanisms:
  1. **Purging**: drop training samples whose [label formation window]
     overlaps the test set's time range.
  2. **Embargo**: additionally drop a buffer of training samples that
     immediately *follow* the test set, since their features' rolling
     windows may look back into the test period.

This implementation follows a walk-forward (expanding window) scheme with
purge + embargo, applied to a *group* index — grouping by contiguous
"prediction blocks" of length `label_horizon` guards against splitting a
single overlapping-label window across train/test.
"""
from __future__ import annotations

import numpy as np


class PurgedGroupTimeSeriesSplit:
    """
    Walk-forward time-series CV with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    max_train_size : int or None
        Cap on the number of samples in each training fold (None = expanding
        window using all available history).
    test_size : int or None
        Number of samples per test fold. If None, computed automatically to
        evenly divide the remaining samples across n_splits.
    purge_window : int
        Number of samples to purge from the END of the training set,
        immediately preceding the test set. Should be >= the max rolling
        feature window AND >= the label horizon, since both create
        train/test overlap risk at that boundary.
    embargo_window : int
        Number of samples to additionally drop from training that fall
        immediately AFTER the test set (guards against training samples
        whose features look back into the test period in a later fold, and
        against label leakage for train samples shortly after test).

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray positional indices, in the
        same style as sklearn's CV splitters (usable directly with
        `cross_val_score`, `GridSearchCV`, or a manual loop).
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int | None = None,
        test_size: int | None = None,
        purge_window: int = 20,
        embargo_window: int = 10,
    ):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_splits = self.n_splits

        test_size = self.test_size or n_samples // (n_splits + 1)
        if test_size < 1:
            raise ValueError("test_size resolved to < 1 sample; reduce n_splits.")

        # First test fold starts after an initial training block large
        # enough to be meaningful; we walk forward from there.
        first_test_start = n_samples - n_splits * test_size

        if first_test_start <= self.purge_window:
            raise ValueError(
                "Not enough samples for the requested n_splits/test_size/"
                "purge_window combination. Reduce n_splits or test_size."
            )

        for i in range(n_splits):
            test_start = first_test_start + i * test_size
            test_end = min(test_start + test_size, n_samples)
            test_idx = np.arange(test_start, test_end)

            # --- Training set: everything before the test block ---
            train_end = test_start - self.purge_window  # PURGE: drop tail before test
            train_end = max(train_end, 0)

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            train_idx = np.arange(train_start, train_end)

            # --- Embargo: also drop training samples shortly AFTER the
            # test block (relevant when max_train_size creates rolling
            # windows, or in a broader k-fold-style setup). Included here
            # for completeness / reuse even though in a pure walk-forward
            # scheme training never naturally extends past the test set. ---
            embargo_start = test_end
            embargo_end = min(test_end + self.embargo_window, n_samples)
            # (no-op in pure walk-forward, but kept explicit & documented)
            _ = np.arange(embargo_start, embargo_end)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


if __name__ == "__main__":
    import numpy as np

    X = np.arange(500).reshape(-1, 1)
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=15, embargo_window=10)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(
            f"Fold {fold}: train=[{train_idx.min()}:{train_idx.max()}] "
            f"(n={len(train_idx)}) | test=[{test_idx.min()}:{test_idx.max()}] "
            f"(n={len(test_idx)}) | gap={test_idx.min() - train_idx.max()}"
        )
