"""Loader OBD — patch pre_process dla pandas 2.x."""

from __future__ import annotations

import pandas as pd
from obp.dataset.real import OpenBanditDataset as _OpenBanditDataset
from sklearn.preprocessing import LabelEncoder


class OpenBanditDataset(_OpenBanditDataset):
    """obp 0.4.1 — drop(columns=...) zamiast deprecated API."""

    def pre_process(self) -> None:
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        item_feature_0 = self.item_context["item_feature_0"].to_frame()
        item_feature_cat = self.item_context.drop(
            columns=["item_id", "item_feature_0"], errors="ignore"
        ).apply(LabelEncoder().fit_transform)
        self.action_context = pd.concat(
            [item_feature_cat, item_feature_0], axis=1
        ).values
