"""The scaler only ever sees training rows: on the final fit and inside every CV fold."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.data import LABEL_COL, deduplicate, stratified_split
from src.features import CLIP_BOUND, apply_scaler, fit_scaler, load_scaler, save_scaler
from src.models import CV_FOLDS, RANDOM_STATE, build_pipeline, honest_cv, logreg_estimator


@pytest.fixture
def split(small_synthetic_df):
    clean, _, _ = deduplicate(small_synthetic_df)
    return stratified_split(clean)


def test_scaler_statistics_come_from_train_only(split):
    X_train, X_test, _, _ = split
    scaler = fit_scaler(X_train)
    np.testing.assert_allclose(scaler.center_, X_train.median().to_numpy())

    # Changing the test rows must not change anything the scaler learned.
    scaler_again = fit_scaler(X_train)
    X_test_shifted = X_test * 100 + 1e6
    apply_scaler(scaler_again, X_test_shifted)
    np.testing.assert_allclose(scaler_again.center_, scaler.center_)
    np.testing.assert_allclose(scaler_again.scale_, scaler.scale_)


def test_scaled_train_is_centered_and_clipped(split):
    X_train, X_test, _, _ = split
    scaler = fit_scaler(X_train)
    train_s = apply_scaler(scaler, X_train)
    test_s = apply_scaler(scaler, X_test * 1_000)

    assert np.abs(train_s.median(axis=0)).max() < 1e-9
    assert train_s.abs().to_numpy().max() <= CLIP_BOUND
    assert test_s.abs().to_numpy().max() == CLIP_BOUND
    assert train_s.index.equals(X_train.index)
    assert list(train_s.columns) == list(X_train.columns)


def test_scaler_roundtrip(tmp_path, split):
    X_train, _, _, _ = split
    scaler = fit_scaler(X_train)
    path = tmp_path / "scaler.pkl"
    save_scaler(scaler, path)
    np.testing.assert_allclose(load_scaler(path).center_, scaler.center_)


def test_cv_pipeline_mirrors_production_preprocessing():
    pipe = build_pipeline(logreg_estimator())
    names = [name for name, _ in pipe.steps]
    assert names == ["scaler", "clip", "clf"]
    assert isinstance(pipe.named_steps["scaler"], RobustScaler)
    assert isinstance(pipe.named_steps["clip"], FunctionTransformer)
    clipped = pipe.named_steps["clip"].transform(np.array([[-50.0, 0.5, 50.0]]))
    np.testing.assert_array_equal(clipped, [[-CLIP_BOUND, 0.5, CLIP_BOUND]])


def test_scaler_is_refit_on_each_training_fold(split):
    """Each fold's scaler must match that fold's training rows, never the full train set."""
    X_train, _, y_train, _ = split
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    out = cross_validate(
        build_pipeline(LogisticRegression(max_iter=500)),
        X_train,
        y_train,
        cv=cv,
        return_estimator=True,
        return_indices=True,
    )
    full_median = X_train.median().to_numpy()
    centers = []
    for est, train_idx, val_idx in zip(
        out["estimator"], out["indices"]["train"], out["indices"]["test"], strict=True
    ):
        assert set(train_idx).isdisjoint(val_idx)
        center = est.named_steps["scaler"].center_
        np.testing.assert_allclose(center, X_train.iloc[train_idx].median().to_numpy())
        assert not np.allclose(center, full_median)
        centers.append(center)
    assert not np.allclose(centers[0], centers[1])


def test_honest_cv_returns_one_macro_f1_per_fold(split):
    X_train, _, y_train, _ = split
    scores = honest_cv(LogisticRegression(max_iter=500), X_train, y_train)
    assert scores.shape == (CV_FOLDS,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores.mean() > 1 / 3  # above chance on data with real class signal


def test_label_column_never_reaches_the_model(small_synthetic_df):
    clean, _, _ = deduplicate(small_synthetic_df)
    X_train, X_test, _, _ = stratified_split(clean)
    assert LABEL_COL not in X_train.columns
    assert LABEL_COL not in X_test.columns
