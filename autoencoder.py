"""Phase 2a: the model. Only this file knows which library is used.
To switch to PyTorch/Keras later, keep these 4 functions and their inputs/outputs."""
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
import config


def train(X, hidden=None):
    """Learn to reproduce X from X through a narrow bottleneck."""
    model = MLPRegressor(hidden_layer_sizes=hidden or config.HIDDEN_LAYERS, activation="tanh",
                         max_iter=config.MAX_ITER, random_state=config.SEED)
    model.fit(X, X)
    # remember each feature's normal error size so no feature gets drowned out
    errs = (np.asarray(X) - model.predict(X)) ** 2
    model.err_scale_ = np.percentile(errs, config.ERROR_SCALE_PERCENTILE, axis=0) + 1e-9
    return model


def train_clean(X, hidden=None):
    """Label-free two-pass training: drop the most unusual rows, retrain on the rest."""
    first = train(X, hidden)
    scores = anomaly_scores(first, X)
    keep = np.asarray(scores <= np.percentile(scores, 100 - config.TRIM_PERCENT))
    model = train(X[keep], hidden)
    model.err_scale_ = np.percentile(reconstruction_errors(model, X), config.ERROR_SCALE_PERCENTILE, axis=0) + 1e-9
    return model, keep


def reconstruction_errors(model, X):
    """Per-feature squared error, shape (rows, features)."""
    return (np.asarray(X) - model.predict(X)) ** 2


def anomaly_scores(model, X):
    """One score per row: worst feature error, relative to normal. High = unusual."""
    return (reconstruction_errors(model, X) / model.err_scale_).max(axis=1)


def save(model):
    joblib.dump(model, config.MODEL_FILE)


def load():
    return joblib.load(config.MODEL_FILE)
