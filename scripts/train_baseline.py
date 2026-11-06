"""Train and evaluate the week-1 baseline."""

from __future__ import annotations

import json

from readmission.data import load
from readmission.evaluate import evaluate
from readmission.features import split_xy, temporal_split
from readmission.model import build_model


def main() -> None:
    df = load()
    train, test = temporal_split(df)

    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    model = build_model(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), probabilities)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
