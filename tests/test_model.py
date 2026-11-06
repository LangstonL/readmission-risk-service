import numpy as np
import pandas as pd

from readmission.evaluate import evaluate
from readmission.model import build_model


def toy_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "age": rng.integers(20, 90, n),
            "race": rng.choice(["A", "B", "C"], n),
            "readmitted": ["<30"] * 20 + ["NO"] * 40,
        }
    )


def test_model_fits_and_predicts_probabilities():
    df = toy_data()
    X = df.drop(columns=["readmitted"])
    y = (df["readmitted"] == "<30").astype(int)

    model = build_model(X, min_frequency=1).fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert probabilities.shape == (60,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_evaluate_returns_expected_keys():
    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert set(metrics) == {"auroc", "auprc", "brier", "prevalence", "n"}
