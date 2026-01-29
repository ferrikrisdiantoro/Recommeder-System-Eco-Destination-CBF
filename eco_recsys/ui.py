
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import streamlit as st
import numpy as np
import pandas as pd

from .state import like_item, skip_item, toggle_bookmark
from .utils import format_idr, get_description

@dataclass
class FeedKnobs:
    top_n: int
    mmr_lambda: float
    per_category_cap: int
    serendipity_pct: int

@dataclass
class FeedbackKnobs:
    use_feedback: bool
    alpha: float
    beta: float
    gamma: float

def sidebar_filters(items: pd.DataFrame):
    all_categories = sorted(set([c.strip()
                                 for s in items["category"].fillna("").tolist()
                                 for c in str(s).split(",") if c.strip()]))
    sel_cats   = st.sidebar.multiselect("Kategori", options=all_categories)
    all_cities = sorted([c for c in items["city"].fillna("").unique() if c])
    sel_cities = st.sidebar.multiselect("Kota/Kabupaten", options=all_cities)
    use_price  = st.sidebar.checkbox("Batasi harga maksimum", value=False)
    max_price_val = float(np.nanmax(items["price"].values)) if items["price"].notna().any() else 0.0
    price_cap  = st.sidebar.slider("Harga Maksimum (IDR)", 0.0, max_price_val,
                                   min(max_price_val, 100_000.0), 1_000.0) if use_price else None
    return {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}

def sidebar_feed_knobs(items: pd.DataFrame) -> FeedKnobs:
    top_n_feed   = st.sidebar.slider("Jumlah item Feed (Top-N)", 5, 40, 12, 1)
    mmr_lambda_f = st.sidebar.slider("MMR λ (Feed)", 0.0, 1.0, 0.7, 0.05)
    per_cat_cap  = st.sidebar.slider("Batas per kategori", 0, 6, 2, 1)
    serendip     = st.sidebar.slider("Serendipity (%)", 0, 30, 15, 5)
    return FeedKnobs(top_n_feed, mmr_lambda_f, per_cat_cap, serendip)

def sidebar_feedback_knobs() -> FeedbackKnobs:
    use_fb = st.sidebar.toggle("Aktifkan reranking Like/Skip (UFW)", value=True)
    alpha = st.sidebar.slider("Boost ke Like (α)", 0.0, 2.0, 0.6, 0.05, disabled=not use_fb)
    beta  = st.sidebar.slider("Penalty Skip (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
    gamma = st.sidebar.slider("Boost kategori Like (γ)", 0.0, 0.2, 0.02, 0.005, disabled=not use_fb)
    return FeedbackKnobs(use_fb, alpha, beta, gamma)

def status_chips():
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)} item")
        with c2: st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)} item")
        with c3: st.markdown(f"**Bookmarks (🔖):** {len(st.session_state.bookmarked_idx)} item")

def search_controls(items: pd.DataFrame):
    st.markdown("**Pencarian Knowledge Base (TF‑IDF):** ketik tema/kata kunci, mis: _pantai aceh_, _snorkeling_, _gunung camping_, dsb.")
    
    # Initialize session state for search if not present
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "search_active" not in st.session_state:
        st.session_state.search_active = False

    # Callback to handle search button click
    def on_search_click():
        st.session_state.search_active = True

    # Input bound to session state (optional, or just read it)
    # Here we use a key to sync with session state automatically if desired, 
    # but to match the original flow, we'll just read the input and update state on click.
    query_input = st.text_input("Kueri pencarian", value=st.session_state.search_query, 
                                placeholder="contoh: snorkeling murah di aceh, hiking keluarga, savana, kebun teh ...",
                                key="search_input_widget")
    
    # Sync input widget to session state query
    st.session_state.search_query = query_input

    top_n_s = st.slider("Jumlah hasil", 5, 40, 12, 1)
    mmr_lambda_s = st.slider("MMR λ (Search)", 0.0, 1.0, 0.7, 0.05)
    min_similarity = st.slider("Min. Similarity (Threshold)", 0.0, 0.5, 0.1, 0.05, 
                               help="Hanya tampilkan hasil dengan kemiripan di atas nilai ini.")
    
    st.button("Cari", type="primary", on_click=on_search_click)
    
    return st.session_state.search_query, top_n_s, mmr_lambda_s, min_similarity, st.session_state.search_active

def render_cards(items: pd.DataFrame, pairs: List[tuple[int,float]], show_score: bool=True, title_suffix: str=""):
    if not pairs:
        st.warning("Tidak ada item untuk konfigurasi saat ini.")
        return
    for gid, sc in pairs:
        row = items.iloc[int(gid)]
        with st.container(border=True):
            cols = st.columns([1, 3])
            with cols[0]:
                img = row.get("place_img")
                if isinstance(img, str) and img.startswith(("http://", "https://")):
                    st.image(img, use_container_width=True)
            with cols[1]:
                st.subheader((row.get("place_name") or "-"))
                st.markdown(f"**Kategori:** {row.get('category') or '-'}  \n**Kota:** {row.get('city') or '-'}")
                rating = row.get("rating")
                price  = row.get("price")
                st.markdown(f"**Rating:** {'-' if pd.isna(rating) else round(float(rating), 2)}  \n**Harga:** {format_idr(None if pd.isna(price) else float(price))}")
                link = row.get("place_map")
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    st.link_button("Buka peta", link, use_container_width=False)
                if show_score:
                    st.caption(f"Skor: {float(sc):.4f}")
                with st.expander("Lihat deskripsi"):
                    st.write(get_description(row))

                b1, b2, b3, _ = st.columns([1,1,1,6])
                gid = int(gid)
                with b1:
                    if st.button("⭐ Suka", key=f"like_{title_suffix}_{gid}"):
                        like_item(gid); st.rerun()
                with b2:
                    if st.button("🚫 Skip", key=f"skip_{title_suffix}_{gid}"):
                        skip_item(gid); st.rerun()
                with b3:
                    label = "Hapus 🔖" if gid in st.session_state.bookmarked_idx else "🔖 Bookmark"
                    if st.button(label, key=f"bm_{title_suffix}_{gid}"):
                        toggle_bookmark(gid); st.rerun()
