import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd

from eco_recsys.data import load_artifacts, ensure_min_columns
from eco_recsys.state import init_session, like_item, skip_item, toggle_bookmark, clear_feedback
from eco_recsys.ui import (
    sidebar_filters, sidebar_feed_knobs, sidebar_feedback_knobs, status_chips,
    render_cards, search_controls
)
from eco_recsys.cbf import build_feed_cbf, search_cbf
from eco_recsys.ufw import apply_ufw

st.set_page_config(page_title="EcoTourism CBF + UFW", page_icon="🌿", layout="wide")

BASE_DIR = Path(__file__).parent
ART_DIR  = BASE_DIR / "artifacts"

# ---------------- Init & Load ----------------
init_session()
with st.spinner("Memuat artifacts..."):
    items, X, vectorizer, nbrs = load_artifacts(ART_DIR)
ensure_min_columns(items)

st.title("🌿 EcoTourism Recommender — CBF + UFW")
st.caption("CBF dasar + User Feedback Weighting | TF‑IDF + NearestNeighbors (cosine) | Like/Skip/Bookmark (session)")

# ---------------- Sidebar (modular UI) ----------------
st.sidebar.header("Filter")
filters = sidebar_filters(items)

st.sidebar.header("Pengaturan Feed")
feed_knobs = sidebar_feed_knobs(items)

st.sidebar.header("Session Feedback (UFW)")
fb_knobs = sidebar_feedback_knobs()

if st.sidebar.button("Reset Like/Skip", type="secondary"):
    clear_feedback()
    st.sidebar.success("Preferensi sesi direset.")

status_chips()

# ---------------- Tabs ----------------
tab_feed, tab_search, tab_book = st.tabs(["🏠 Feed", "🔎 Search / KB", "🔖 Bookmarks"])

# ---------------- FEED (CBF dasar + UFW) ----------------
with tab_feed:
    cbf_candidates = build_feed_cbf(
        items=items, X=X, filters=filters,
        top_n=feed_knobs.top_n,
        mmr_lambda=feed_knobs.mmr_lambda,
        per_category_cap=feed_knobs.per_category_cap,
        serendipity_pct=feed_knobs.serendipity_pct
    )
    # Apply UFW reranking if enabled
    if fb_knobs.use_feedback and cbf_candidates:
        final_pairs = apply_ufw(
            gids=[cid.gid for cid in cbf_candidates],
            base_scores_map={cid.gid: cid.base_score for cid in cbf_candidates},
            X=X, items=items,
            alpha=fb_knobs.alpha, beta=fb_knobs.beta, gamma=fb_knobs.gamma
        )
    else:
        final_pairs = [(cid.gid, cid.base_score) for cid in cbf_candidates]

    render_cards(items, final_pairs, title_suffix="Feed")

# ---------------- SEARCH / KB (CBF query) ----------------
with tab_search:
    query, top_n_s, mmr_lambda_s, do_search = search_controls(items)
    if do_search and query.strip():
        results = search_cbf(
            items=items, X=X, vectorizer=vectorizer, nbrs=nbrs,
            query=query, filters=filters,
            top_n=top_n_s, mmr_lambda=mmr_lambda_s,
            per_category_cap=feed_knobs.per_category_cap
        )
        # UFW
        if fb_knobs.use_feedback and results:
            final_pairs = apply_ufw(
                gids=[gid for gid,_ in results],
                base_scores_map=dict(results),
                X=X, items=items,
                alpha=fb_knobs.alpha, beta=fb_knobs.beta, gamma=fb_knobs.gamma
            )
        else:
            final_pairs = results
        render_cards(items, final_pairs, title_suffix="Search/KB")
    else:
        st.info("Masukkan kueri lalu klik **Cari** untuk melihat hasil.")

# ---------------- BOOKMARKS ----------------
with tab_book:
    bms = sorted(list(st.session_state.bookmarked_idx))
    if not bms:
        st.info("Belum ada item yang di‑bookmark.")
    else:
        render_cards(items, [(gid, 0.0) for gid in bms], show_score=False, title_suffix="Bookmarks")

st.write("---")
st.caption("Note: eco_recsys.cbf (CBF), eco_recsys.ufw (User Feedback Weighting), eco_recsys.ui (UI), eco_recsys.data (loader)")
