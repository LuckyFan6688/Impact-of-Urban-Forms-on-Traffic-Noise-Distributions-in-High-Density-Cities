"""Nominal four-pattern XGBoost classification and task-specific metrics.

Classes 0..3 correspond to T1 high-value attenuation, T2 medium-value
attenuation, T3 low-value attenuation, and T4 low total attenuation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss)
from xgboost import XGBClassifier

CLASSES = [0, 1, 2, 3]


def fit_classifier(x, y, cfg):
    """Every training fold must contain all four nominal classes."""
    if sorted(np.unique(y).tolist()) != CLASSES:
        raise ValueError('Every training fold must contain all four nominal classes (0 through 3).')
    params = dict(cfg['models']['Classification'], random_state=cfg['random_seed'],
                  n_jobs=cfg['n_jobs'], tree_method='hist')
    return XGBClassifier(**params, objective='multi:softprob', num_class=4).fit(x, y)


def classification_metrics(y, probability):
    probability = np.asarray(probability)
    if (probability.ndim != 2 or probability.shape[1] != 4
        or not np.isfinite(probability).all() or np.any(probability < 0)
        or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=5e-6)):
        raise ValueError('Four finite probabilities summing to one are required.')
    prediction = np.argmax(probability, axis=1)
    return {'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'macro_F1': float(f1_score(y, prediction, labels=CLASSES, average='macro', zero_division=0)),
            'log_loss': float(log_loss(y, probability, labels=CLASSES)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=CLASSES).tolist()}
