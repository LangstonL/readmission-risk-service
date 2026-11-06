import pandas as pd

from readmission.features import make_target, split_xy, temporal_split


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "patient_nbr": [10, 10, 11, 12, 13],
            "time_in_hospital": [1, 2, 3, 4, 5],
            "race": ["Caucasian", None, "AfricanAmerican", "Caucasian", "Other"],
            "discharge_disposition_id": [1, 1, 11, 1, 1],
            "readmitted": ["NO", ">30", "<30", "NO", ">30"],
        }
    )


def test_target_is_30_day_readmission():
    assert make_target(sample_df()).tolist() == [0, 0, 1, 0, 0]


def test_split_drops_ids_leaky_columns_and_target():
    X, y = split_xy(sample_df())
    for column in ["encounter_id", "patient_nbr", "discharge_disposition_id", "readmitted"]:
        assert column not in X.columns
    assert y.tolist() == [0, 0, 1, 0, 0]


def test_temporal_split_is_ordered_and_disjoint():
    train, test = temporal_split(sample_df(), test_fraction=0.4)
    assert len(train) == 3
    assert len(test) == 2
    assert train["encounter_id"].max() < test["encounter_id"].min()
