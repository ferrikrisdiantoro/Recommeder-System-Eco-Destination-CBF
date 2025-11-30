
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .text import preprocess_text
from .utils import normalize_minmax

@dataclass
class Candidate:
    gid: int
    base_score: float

def _mask_by_filters(items: pd.DataFrame, filters) -> np.ndarray:
    mask = np.full(len(items), True, dtype=bool)
    if filters.get("categories"):
        cats = [c.lower() for c in filters["categories"]]
        mask &= items["category"].fillna("").apply(lambda s: any(c in s.lower() for c in cats)).values
    if filters.get("cities"):
        cts = [c.lower() for c in filters["cities"]]
        mask &= items["city"].fillna("").apply(lambda s: s.lower() in cts).values
    if filters.get("max_price") is not None:
        p = items["price"].fillna(np.inf).values.astype(float)
        mask &= p <= float(filters["max_price"])
    return mask

def mmr_select(idx_all, X, base_scores, top_n=20, lambda_mmr=0.7,
               per_category_cap=0, items: Optional[pd.DataFrame] = None):
    selected_loc: list[int] = []
    candidates = list(range(len(idx_all)))
    cat_count = {}

    while candidates and len(selected_loc) < min(top_n * 3, len(idx_all)):
        best_loc, best_val = None, -1e18
        for loc in candidates:
            gid = int(idx_all[loc])
            if per_category_cap and items is not None:
                cat = str(items.iloc[gid]["category"]).split(",")[0].strip()
                if cat and cat_count.get(cat, 0) >= per_category_cap:
                    continue

            score = float(base_scores[loc])
            if not selected_loc:
                mmr = score
            else:
                sel_g = idx_all[selected_loc]
                sim = cosine_similarity(X[gid], X[sel_g]).max()
                mmr = lambda_mmr * score - (1.0 - lambda_mmr) * float(sim)

            if mmr > best_val:
                best_val, best_loc = mmr, loc

        if best_loc is None:
            break
        selected_loc.append(best_loc)
        if per_category_cap and items is not None:
            cat = str(items.iloc[int(idx_all[best_loc])]["category"]).split(",")[0].strip()
            if cat:
                cat_count[cat] = cat_count.get(cat, 0) + 1
        candidates.remove(best_loc)
        if len(selected_loc) >= top_n:
            break

    return [int(idx_all[loc]) for loc in selected_loc]

def build_feed_cbf(items, X, filters, top_n=12, mmr_lambda=0.7, per_category_cap=2, serendipity_pct=15, blocked_gids=None):
    mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[mask]

    # Filter out blocked items if any
    if blocked_gids:
        blocked_set = set(blocked_gids)
        idx_all = np.array([i for i in idx_all if i not in blocked_set])

    if idx_all.size == 0:
        return []

    rating = items.iloc[idx_all]["rating"].fillna(0.0).clip(lower=0.0).values.astype(float)
    base_scores = normalize_minmax(rating) + np.random.RandomState(13).rand(len(idx_all)) * 1e-4

    gids = mmr_select(idx_all, X, base_scores, top_n=top_n,
                      lambda_mmr=mmr_lambda, per_category_cap=per_category_cap, items=items)

    selected = set(gids)
    ser_k = max(0, min(max(1, top_n // 5), int(len(idx_all) * serendipity_pct / 100)))
    if ser_k > 0:
        pool = [g for g in idx_all if g not in selected]
        if pool:
            top_pop = items.iloc[pool].copy()
            pool_sorted = list(top_pop.sort_values(["rating"], ascending=False).index)
            rng = np.random.RandomState(31)
            rng.shuffle(pool_sorted)
            pick = pool_sorted[:ser_k]
            gids.extend(pick)

    base_map = {int(idx_all[i]): float(base_scores[i]) for i in range(len(idx_all))}
    out = [Candidate(gid=int(g), base_score=float(base_map.get(int(g), 0.0))) for g in gids[:top_n]]
    return out

def search_cbf(items, X, vectorizer, nbrs, query: str, filters, top_n=12, mmr_lambda=0.7, per_category_cap=3):
    if vectorizer is None:
        return []

    mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[mask]
    if idx_all.size == 0:
        return []

    qv = vectorizer.transform([preprocess_text(query)])

    if nbrs is not None:
        dist, neigh_idx = nbrs.kneighbors(qv, n_neighbors=min(max(100, top_n*5), len(items)))
        candidates = neigh_idx[0]
        sims_full = 1.0 - dist[0]
        mask_set = set(idx_all.tolist())
        sel = [(int(g), float(s)) for g, s in zip(candidates, sims_full) if g in mask_set]
        if not sel:
            sims = cosine_similarity(X[idx_all], qv)[:, 0]
            base_scores = normalize_minmax(sims)
            gids = mmr_select(idx_all, X, base_scores, top_n=top_n,
                              lambda_mmr=mmr_lambda, per_category_cap=per_category_cap, items=items)
            return [(int(g), float(base_scores[list(idx_all).index(int(g))])) for g in gids]
        sub_gids = np.array([g for g,_ in sel], dtype=int)
        sub_scores = normalize_minmax([s for _,s in sel])
        gids = mmr_select(sub_gids, X, sub_scores, top_n=top_n,
                          lambda_mmr=mmr_lambda, per_category_cap=per_category_cap, items=items)
        base_map = {int(g): float(s) for g,s in zip(sub_gids, sub_scores)}
        return [(int(g), float(base_map.get(int(g), 0.0))) for g in gids]
    else:
        sims = cosine_similarity(X[idx_all], qv)[:, 0]
        base_scores = normalize_minmax(sims)
        gids = mmr_select(idx_all, X, base_scores, top_n=top_n,
                          lambda_mmr=mmr_lambda, per_category_cap=per_category_cap, items=items)
        return [(int(g), float(base_scores[list(idx_all).index(int(g))])) for g in gids]
