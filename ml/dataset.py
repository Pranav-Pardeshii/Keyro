"""
CMU Keystroke Dynamics Benchmark Dataset Loader and Splitter.
"""

import os
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "DSL-StrongPasswordData.csv"
)

FEATURE_COLUMNS = [
    "H.period", "DD.period.t", "UD.period.t",
    "H.t", "DD.t.i", "UD.t.i",
    "H.i", "DD.i.e", "UD.i.e",
    "H.e", "DD.e.five", "UD.e.five",
    "H.five", "DD.five.Shift.r", "UD.five.Shift.r",
    "H.Shift.r", "DD.Shift.r.o", "UD.Shift.r.o",
    "H.o", "DD.o.a", "UD.o.a",
    "H.a", "DD.a.n", "UD.a.n",
    "H.n", "DD.n.l", "UD.n.l",
    "H.l", "DD.l.Return", "UD.l.Return",
    "H.Return"
]

METADATA_COLUMNS = ["subject", "sessionIndex", "rep"]


def load_cmu_dataset(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads and validates the CMU Keystroke Benchmark dataset.
    If the file does not exist, attempts to download it automatically.
    """
    if not os.path.exists(file_path):
        from data.download_dataset import download_dataset
        download_dataset(file_path)

    df = pd.read_csv(file_path)

    # Validate columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {file_path} is missing expected feature columns: {missing}")

    return df


def get_all_subjects(df: pd.DataFrame) -> List[str]:
    """Returns a sorted list of unique subjects in the dataset."""
    return sorted(df["subject"].unique().tolist())


def get_user_splits(
    df: pd.DataFrame,
    subject: str,
    train_reps: int = 200,
    impostors_per_other_user: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions the dataset according to the Killourhy & Maxion (2009) benchmark protocol:
    - Train set: First `train_reps` repetitions of the target genuine user.
    - Genuine test set: Remaining repetitions (e.g. reps 201-400) of the target genuine user.
    - Impostor test set: `impostors_per_other_user` samples from each of the other subjects' test sets.
      For 51 subjects, 50 other subjects * 5 = 250 impostor samples.

    Returns:
        (X_train, X_test_genuine, X_test_impostor) as float64 numpy arrays of shape (N, 31).
    """
    sub_df = df[df["subject"] == subject].sort_values(by=["sessionIndex", "rep"])
    if len(sub_df) < train_reps:
        raise ValueError(f"Subject {subject} has only {len(sub_df)} rows, less than {train_reps}")

    train_df = sub_df.iloc[:train_reps]
    test_gen_df = sub_df.iloc[train_reps:]

    other_df = df[df["subject"] != subject]
    
    # Take test reps from other users (rows after train_reps)
    impostor_rows = []
    for other_subj, group in other_df.groupby("subject"):
        sorted_g = group.sort_values(by=["sessionIndex", "rep"])
        test_portion = sorted_g.iloc[train_reps:]
        sample = test_portion.iloc[:impostors_per_other_user]
        impostor_rows.append(sample)

    impostors_df = pd.concat(impostor_rows, axis=0)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X_test_gen = test_gen_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X_test_imp = impostors_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    return X_train, X_test_gen, X_test_imp
