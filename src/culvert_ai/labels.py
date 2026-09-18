from __future__ import annotations

import pandas as pd


TRAINING_LABEL_COLUMN = "training_label_class"
VERIFIED_POSITIVE = "verified_positive"
VERIFIED_NEGATIVE = "verified_negative"
WEAK_NEGATIVE = "weak_negative"
UNLABELED_CANDIDATE = "unlabeled_candidate"

TRAINING_LABEL_CLASSES = (
    VERIFIED_POSITIVE,
    VERIFIED_NEGATIVE,
    WEAK_NEGATIVE,
    UNLABELED_CANDIDATE,
)


def training_label_classes(table: pd.DataFrame, target_column: str = "is_culvert") -> pd.Series:
    """Return explicit classes, conservatively upgrading legacy feature tables.

    Legacy zero-valued targets are never treated as negatives. Only positive targets and
    field-denial metadata can create a training label when the explicit class column is absent.
    """

    if TRAINING_LABEL_COLUMN in table.columns:
        labels = table[TRAINING_LABEL_COLUMN].fillna(UNLABELED_CANDIDATE).astype(str)
        invalid = sorted(set(labels) - set(TRAINING_LABEL_CLASSES))
        if invalid:
            raise ValueError(f"Unknown training label classes: {', '.join(invalid)}")
        return labels

    labels = pd.Series(UNLABELED_CANDIDATE, index=table.index, dtype=object)
    if target_column in table.columns:
        positive = pd.to_numeric(table[target_column], errors="coerce").fillna(0).astype(int).eq(1)
        labels.loc[positive] = VERIFIED_POSITIVE

    if "field_denied" not in table.columns:
        return labels

    denied = (
        pd.to_numeric(table["field_denied"], errors="coerce").fillna(0).astype(int).eq(1)
        & labels.ne(VERIFIED_POSITIVE)
    )
    candidate_source = _text_column(table, "source")
    direct_label = _text_column(table, "field_observation_label")
    direct_no_culvert = (
        denied & candidate_source.eq("field_observed_non_culvert") & direct_label.eq("no_culvert")
    )
    labels.loc[denied] = WEAK_NEGATIVE
    labels.loc[direct_no_culvert] = VERIFIED_NEGATIVE
    return labels


def _text_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series("", index=table.index, dtype=object)
    return table[column].fillna("").astype(str).str.strip().str.lower()
