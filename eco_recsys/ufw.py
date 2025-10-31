
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from .utils import normalize_minmax

def _compute_centroid_dense(X, indices) -> np.ndarray | None:
    if not indices:
        return None
    rows = X[list(indices)]
    mean_mat = rows.mean(axis=0)
    cent = np.asarray(getattr(mean_mat, "A", mean_mat)).reshape(1, -1)
    return cent

def apply_ufw(gids: List[int], base_scores_map: Dict[int, float], X, items: pd.DataFrame,
              alpha: float=0.6, beta: float=0.7, gamma: float=0.02) -> List[Tuple[int,float]]:
    liked   = list(st.session_state.liked_idx)
    blocked = set(st.session_state.blocked_idx)

    cat_pref: dict[str, int] = {}
    if liked:
        liked_cats = (items.iloc[liked]["category"].fillna("")
                      .apply(lambda s: str(s).split(",")[0].strip()))
        for c in liked_cats:
            if c:
                cat_pref[c] = cat_pref.get(c, 0) + 1

    cent = _compute_centroid_dense(X, liked)
    sims = np.zeros(len(gids), dtype=float)
    if cent is not None and len(gids):
        sims = cosine_similarity(X[gids], cent)[:, 0]
        sims = normalize_minmax(sims)

    new_scores = {}
    for i, gid in enumerate(gids):
        base = float(base_scores_map.get(gid, 0.0))
        s = base + (alpha * sims[i] if liked else 0.0)
        if gid in blocked:
            s -= beta
        cat = str(items.iloc[gid]["category"]).split(",")[0].strip()
        if cat and cat_pref:
            s += gamma * cat_pref.get(cat, 0)
        new_scores[gid] = s

    return sorted(new_scores.items(), key=lambda x: x[1], reverse=True)
