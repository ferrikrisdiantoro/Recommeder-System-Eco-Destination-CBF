import streamlit as st
import pandas as pd
import numpy as np
import joblib, random, math, re
from pathlib import Path
from typing import Optional
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="EcoTourism CBF", page_icon="🌿", layout="wide")
random.seed(13)

# ====================== Lokasi artefak dari notebook ======================
BASE_DIR = Path(__file__).parent
ART_DIR  = BASE_DIR / "artifacts"

@st.cache_resource(show_spinner=True)
def load_artifacts():
    """
    Memuat artefak:
      - items.csv        : metadata item + kolom 'gabungan' (teks terproses)
      - tfidf_matrix.npz : matriks fitur item (CSR sparse)
      - vectorizer.joblib: TF-IDF vectorizer (untuk Search/KB)
    """
    items = pd.read_csv(ART_DIR / "items.csv")
    X     = load_npz(ART_DIR / "tfidf_matrix.npz").tocsr()
    try:
        vectorizer = joblib.load(ART_DIR / "vectorizer.joblib")
    except Exception:
        vectorizer = None  # Search/KB akan disable bila None

    # Pastikan kolom UI minimal tersedia
    for col in ["place_name", "category", "city", "price", "rating",
                "place_img", "place_map", "gabungan"]:
        if col not in items.columns:
            items[col] = np.nan

    return items, X, vectorizer

# ============================== Util tampilan ==============================
def format_idr(x):
    """Format angka ke Rupiah, aman untuk NaN/inf."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return "Rp{:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return str(x)

def get_description(row: pd.Series) -> str:
    """
    Ambil deskripsi yang human-readable:
    - gunakan place_description jika ada & tidak kosong,
    - jika tidak ada, fallback ke kolom 'gabungan' (teks terproses).
    """
    desc = row.get("place_description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return str(row.get("gabungan") or "")

# ============================ Session state ================================
def init_session():
    # Simpan indeks global item yang di-*Like*/di-*Skip*/di-*Bookmark* selama sesi
    st.session_state.setdefault("liked_idx", set())
    st.session_state.setdefault("blocked_idx", set())
    st.session_state.setdefault("bookmarked_idx", set())
init_session()

def like_item(gid: int):
    st.session_state.liked_idx.add(int(gid))
    st.session_state.blocked_idx.discard(int(gid))

def skip_item(gid: int):
    st.session_state.blocked_idx.add(int(gid))
    st.session_state.liked_idx.discard(int(gid))

def toggle_bookmark(gid: int):
    gid = int(gid)
    if gid in st.session_state.bookmarked_idx:
        st.session_state.bookmarked_idx.remove(gid)
    else:
        st.session_state.bookmarked_idx.add(gid)

def clear_feedback():
    st.session_state.liked_idx.clear()
    st.session_state.blocked_idx.clear()

# ============================ Util numerik =================================
def normalize_minmax(a: np.ndarray) -> np.ndarray:
    """Normalisasi min–max; jika rentang nol -> vektor nol."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

def compute_centroid_dense(X, indices: list[int] | set[int] | tuple[int]) -> Optional[np.ndarray]:
    """
    Centroid dari baris-baris X[indices] sebagai ndarray (1, d).
    Penting: .mean() di sparse mengembalikan numpy.matrix -> konversi via .A / np.asarray.
    """
    if not indices:
        return None
    rows = X[list(indices)]                 # CSR (k, d)
    mean_mat = rows.mean(axis=0)            # numpy.matrix (1, d)
    cent = np.asarray(getattr(mean_mat, "A", mean_mat)).reshape(1, -1)
    return cent

# ============================ Diversifikasi (MMR) ==========================
def mmr_select(idx_all, X, base_scores, top_n=20, lambda_mmr=0.7,
               per_category_cap=0, items: Optional[pd.DataFrame] = None):
    """
    Pilih set kandidat beragam via MMR:
      argmax  λ * base_score  −  (1 − λ) * max_sim_ke_item_terpilih
    """
    selected_loc: list[int] = []
    candidates = list(range(len(idx_all)))
    cat_count = {}

    while candidates and len(selected_loc) < min(top_n * 3, len(idx_all)):
        best_loc, best_val = None, -1e18
        for loc in candidates:
            gid = int(idx_all[loc])

            # Batasi jumlah per kategori (opsional)
            if per_category_cap and items is not None:
                cat = str(items.iloc[gid]["category"]).split(",")[0].strip()
                if cat and cat_count.get(cat, 0) >= per_category_cap:
                    continue

            score = float(base_scores[loc])
            if not selected_loc:
                mmr = score
            else:
                sel_g = idx_all[selected_loc]                      # (m,)
                sim = cosine_similarity(X[gid], X[sel_g]).max()    # -> float
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

# ============================ Reranking feedback ===========================
def rerank_with_feedback(gids, base_scores_map, X, items,
                         alpha=0.6, beta=0.7, gamma=0.02):
    """
    Skor akhir = base_score
               + alpha * sim(item, centroid_like)
               - beta  * 1[item di blocked]
               + gamma * preferensi kategori dari Like
    """
    liked   = list(st.session_state.liked_idx)
    blocked = set(st.session_state.blocked_idx)

    # Preferensi kategori dari Like
    cat_pref: dict[str, int] = {}
    if liked:
        liked_cats = (items.iloc[liked]["category"].fillna("")
                      .apply(lambda s: str(s).split(",")[0].strip()))
        for c in liked_cats:
            if c:
                cat_pref[c] = cat_pref.get(c, 0) + 1

    # Centroid Like -> ndarray (1, d)
    cent = compute_centroid_dense(X, liked)
    sims = np.zeros(len(gids), dtype=float)
    if cent is not None and len(gids):
        sims = cosine_similarity(X[gids], cent)[:, 0]
        sims = normalize_minmax(sims)

    # Terapkan boost/penalti
    new_scores = {}
    for i, gid in enumerate(gids):
        base = float(base_scores_map.get(gid, 0.0))
        s = base
        s += (alpha * sims[i]) if liked else 0.0
        if gid in blocked:
            s -= beta
        cat = str(items.iloc[gid]["category"]).split(",")[0].strip()
        if cat and cat_pref:
            s += gamma * cat_pref.get(cat, 0)
        new_scores[gid] = s

    return sorted(new_scores.items(), key=lambda x: x[1], reverse=True)

# ================================ FEED =====================================
def build_feed(items, X, filters, top_n=12, mmr_lambda=0.7, per_cat_cap=2,
               serendipity_pct=15, use_feedback=True,
               alpha=0.6, beta=0.7, gamma=0.02):
    """
    Feed tanpa query:
    1) Filter -> idx_all
    2) Base score = rating ternormalisasi (+ noise kecil)
    3) Diversifikasi via MMR
    4) Serendipity: sisipkan beberapa top-rated di luar kandidat
    5) Rerank dengan Like/Skip
    """
    # 1) Filter
    mask = np.full(len(items), True, dtype=bool)
    if filters["categories"]:
        cats = [c.lower() for c in filters["categories"]]
        mask &= items["category"].fillna("").apply(lambda s: any(c in s.lower() for c in cats)).values
    if filters["cities"]:
        cts = [c.lower() for c in filters["cities"]]
        mask &= items["city"].fillna("").apply(lambda s: s.lower() in cts).values
    if filters["max_price"] is not None:
        p = items["price"].fillna(np.inf).values.astype(float)
        mask &= p <= float(filters["max_price"])

    idx_all = np.arange(len(items))[mask]
    if idx_all.size == 0:
        return []

    # 2) Base score (proxy populer: rating)
    rating = items.iloc[idx_all]["rating"].fillna(0.0).clip(lower=0.0).values.astype(float)
    base_scores = normalize_minmax(rating) + np.random.RandomState(13).rand(len(idx_all)) * 1e-4

    # 3) Diversifikasi via MMR
    gids = mmr_select(idx_all, X, base_scores, top_n=top_n,
                      lambda_mmr=mmr_lambda, per_category_cap=per_cat_cap, items=items)

    # Peta skor dasar -> indeks global
    base_map = {int(idx_all[i]): float(base_scores[i]) for i in range(len(idx_all))}

    # 4) Serendipity
    selected = set(gids)
    ser_k = max(0, min(max(1, top_n // 5), int(len(idx_all) * serendipity_pct / 100)))
    if ser_k > 0:
        pool = [g for g in idx_all if g not in selected]
        if pool:
            top_pop = items.iloc[pool].copy()
            pool_sorted = list(top_pop.sort_values(["rating"], ascending=False).index)
            random.shuffle(pool_sorted)
            pick = pool_sorted[:ser_k]
            gids.extend(pick)
            for g in pick:
                base_map[int(g)] = min(base_map.get(int(g), 0.0), 0.0)  # tandai serendipity

    # 5) Rerank dengan feedback sesi
    if use_feedback:
        pairs = rerank_with_feedback(gids, base_map, X, items, alpha=alpha, beta=beta, gamma=gamma)
    else:
        pairs = sorted([(g, base_map.get(g, 0.0)) for g in gids], key=lambda x: x[1], reverse=True)

    # 6) Kemasan output
    out = []
    for gid, sc in pairs[:top_n]:
        row = items.iloc[int(gid)]
        out.append({
            "gid": int(gid),
            "place_name": row.get("place_name"),
            "place_img": row.get("place_img"),
            "place_map": row.get("place_map"),
            "category": row.get("category"),
            "city": row.get("city"),
            "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
            "price": None if pd.isna(row.get("price")) else float(row.get("price")),
            "description": get_description(row),
            "score": float(sc),
        })
    return out

# ============================== SEARCH / KB ================================
# Preprocess ringan untuk query
STOPWORDS_ID = {
    "yang","untuk","kepada","terhadap","dapat","para","tanpa","bukan","oleh","saat","kami","kamu","mereka",
    "sebagai","adalah","itu","ini","ada","atau","dan","dengan","dari","di","ke","pada","dalam","serta","agar",
    "akan","bila","jika","supaya","karena","tentang","yaitu","yakni","juga","namun","tapi","hanya","saja",
    "lebih","sudah","belum","pernah","sangat","masih","pun","lah","kah","nya","harus","bisa","tentu","mungkin",
    "lalu","kemudian","hingga","sampai","antara","suatu","sebuah","tiap","setiap","banyak","semua","seluruh",
    "berbagai","beberapa","lainnya","lain"
}
IMPORTANT_WORDS = {"di","ke","dari","untuk","dengan","yang"}  # kata depan penting dipertahankan

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    filtered = [t for t in tokens if (t not in STOPWORDS_ID) or (t in IMPORTANT_WORDS)]
    return " ".join(filtered)

def build_search(items, X, vectorizer, query: str, filters,
                 top_n=12, mmr_lambda=0.7, per_cat_cap=3,
                 use_feedback=True, alpha=0.6, beta=0.7, gamma=0.02):
    """
    Search/KB:
    1) Transform query -> tf-idf (pakai vectorizer dari notebook)
    2) Hitung cosine ke seluruh item (atau subset sesuai filter)
    3) Diversifikasi via MMR
    4) Rerank dengan Like/Skip
    """
    if vectorizer is None:
        return [], "Vectorizer tidak ditemukan. Jalankan notebook untuk menghasilkan artifacts."

    # 1) Filter subset
    mask = np.full(len(items), True, dtype=bool)
    if filters["categories"]:
        cats = [c.lower() for c in filters["categories"]]
        mask &= items["category"].fillna("").apply(lambda s: any(c in s.lower() for c in cats)).values
    if filters["cities"]:
        cts = [c.lower() for c in filters["cities"]]
        mask &= items["city"].fillna("").apply(lambda s: s.lower() in cts).values
    if filters["max_price"] is not None:
        p = items["price"].fillna(np.inf).values.astype(float)
        mask &= p <= float(filters["max_price"])

    idx_all = np.arange(len(items))[mask]
    if idx_all.size == 0:
        return [], None

    # 2) Query -> TF-IDF -> cosine similarity
    q = vectorizer.transform([preprocess_text(query)])
    sims = cosine_similarity(X[idx_all], q)[:, 0]                   # (m,)
    base_scores = normalize_minmax(sims)

    # 3) Diversifikasi MMR (pakai skor sim sebagai base)
    gids = mmr_select(idx_all, X, base_scores, top_n=top_n,
                      lambda_mmr=mmr_lambda, per_category_cap=per_cat_cap, items=items)

    # Map base score ke indeks global
    base_map = {int(idx_all[i]): float(base_scores[i]) for i in range(len(idx_all))}

    # 4) Rerank dengan Like/Skip
    if use_feedback:
        pairs = rerank_with_feedback(gids, base_map, X, items, alpha=alpha, beta=beta, gamma=gamma)
    else:
        pairs = sorted([(g, base_map.get(g, 0.0)) for g in gids], key=lambda x: x[1], reverse=True)

    # 5) Kemasan hasil
    out = []
    for gid, sc in pairs[:top_n]:
        row = items.iloc[int(gid)]
        out.append({
            "gid": int(gid),
            "place_name": row.get("place_name"),
            "place_img": row.get("place_img"),
            "place_map": row.get("place_map"),
            "category": row.get("category"),
            "city": row.get("city"),
            "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
            "price": None if pd.isna(row.get("price")) else float(row.get("price")),
            "description": get_description(row),
            "score": float(sc),  # di tab Search/KB, skor = kombinasi (setelah rerank)
            "similarity": float(sims[list(idx_all).index(gid)]) if gid in idx_all else None,
        })
    return out, None

# ================================== UI ====================================
st.title("🌿 EcoTourism Recommender — CBF Only")
st.caption("Feed otomatis + Search/KB (TF-IDF) • Like/Skip/Bookmark (session) • Tanpa login")

# Muat artefak
try:
    items, X, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artifacts: {e}")
    st.stop()

# Sidebar: filter & knobs
st.sidebar.header("Filter")
all_categories = sorted(set([c.strip()
                             for s in items["category"].fillna("").tolist()
                             for c in str(s).split(",") if c.strip()]))
sel_cats   = st.sidebar.multiselect("Kategori", options=all_categories)
all_cities = sorted([c for c in items["city"].fillna("").unique() if c])
sel_cities = st.sidebar.multiselect("Kota/Kabupaten", options=all_cities)

use_price     = st.sidebar.checkbox("Batasi harga maksimum", value=False)
max_price_val = float(np.nanmax(items["price"].values)) if items["price"].notna().any() else 0.0
price_cap     = st.sidebar.slider("Harga Maksimum (IDR)", 0.0, max_price_val,
                                  min(max_price_val, 100_000.0), 1_000.0) if use_price else None

st.sidebar.header("Pengaturan Feed")
top_n_feed   = st.sidebar.slider("Jumlah item Feed (Top-N)", 5, 40, 12, 1)
mmr_lambda_f = st.sidebar.slider("MMR λ (Feed)", 0.0, 1.0, 0.7, 0.05)
per_cat_cap  = st.sidebar.slider("Batas per kategori", 0, 6, 2, 1)
serendip     = st.sidebar.slider("Serendipity (%)", 0, 30, 15, 5)

st.sidebar.header("Session Feedback")
use_fb = st.sidebar.toggle("Aktifkan reranking Like/Skip", value=True)
alpha = st.sidebar.slider("Boost ke Like (α)", 0.0, 2.0, 0.6, 0.05, disabled=not use_fb)
beta  = st.sidebar.slider("Penalty Skip (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
gamma = st.sidebar.slider("Boost kategori Like (γ)", 0.0, 0.2, 0.02, 0.005, disabled=not use_fb)
if st.sidebar.button("Reset Like/Skip", type="secondary"):
    clear_feedback()
    st.sidebar.success("Preferensi sesi direset.")

# Status chips
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)} item")
    with c2: st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)} item")
    with c3: st.markdown(f"**Bookmarks (🔖):** {len(st.session_state.bookmarked_idx)} item")

# Tabs: Feed • Search/KB • Bookmarks
tab_feed, tab_search, tab_book = st.tabs(["🏠 Feed", "🔎 Search / KB", "🔖 Bookmarks"])

# ------------------------------ FEED ---------------------------------------
with tab_feed:
    filters = {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}
    feed = build_feed(items, X, filters, top_n=top_n_feed, mmr_lambda=mmr_lambda_f,
                      per_cat_cap=per_cat_cap, serendipity_pct=serendip, use_feedback=use_fb,
                      alpha=alpha, beta=beta, gamma=gamma)

    if not feed:
        st.warning("Tidak ada item untuk filter saat ini. Coba longgarkan filter.")
    else:
        for r in feed:
            gid = int(r["gid"])
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if isinstance(r["place_img"], str) and r["place_img"].startswith(("http://", "https://")):
                        st.image(r["place_img"], width='stretch')
                with cols[1]:
                    st.subheader(r["place_name"] or "-")
                    st.markdown(f"**Kategori:** {r['category'] or '-'}  \n**Kota:** {r['city'] or '-'}")
                    st.markdown(f"**Rating:** {'-' if r['rating'] is None else round(r['rating'], 2)}  \n**Harga:** {format_idr(r['price'])}")
                    if isinstance(r["place_map"], str) and r["place_map"].startswith(("http://", "https://")):
                        st.link_button("Buka peta", r["place_map"], width='content')
                    st.caption(f"Skor (setelah rerank): {r['score']:.4f}")

                    with st.expander("Lihat deskripsi"):
                        st.write(r.get("description") or "-")

                    # Actions
                    b1, b2, b3, _ = st.columns([1,1,1,6])
                    with b1:
                        if st.button("⭐ Suka", key=f"like_feed_{gid}"):
                            like_item(gid); st.rerun()
                    with b2:
                        if st.button("🚫 Skip", key=f"skip_feed_{gid}"):
                            skip_item(gid); st.rerun()
                    with b3:
                        label = "Hapus 🔖" if gid in st.session_state.bookmarked_idx else "🔖 Bookmark"
                        if st.button(label, key=f"bm_feed_{gid}"):
                            toggle_bookmark(gid); st.rerun()

# --------------------------- SEARCH / KB -----------------------------------
with tab_search:
    st.markdown("**Pencarian Knowledge Base (TF-IDF):** ketik tema/kata kunci, mis: _pantai aceh_, _snorkeling_, _gunung camping_, dsb.")
    query = st.text_input("Kueri pencarian", placeholder="contoh: snorkeling murah di aceh, hiking keluarga, savana, kebun teh ...")
    top_n_s = st.slider("Jumlah hasil", 5, 40, 12, 1)
    mmr_lambda_s = st.slider("MMR λ (Search)", 0.0, 1.0, 0.7, 0.05)

    if st.button("Cari", type="primary"):
        filters = {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}
        results, err = build_search(items, X, vectorizer, query, filters,
                                    top_n=top_n_s, mmr_lambda=mmr_lambda_s,
                                    per_cat_cap=per_cat_cap, use_feedback=use_fb,
                                    alpha=alpha, beta=beta, gamma=gamma)
        if err:
            st.error(err)
        elif not results:
            st.warning("Tidak ada hasil untuk kueri & filter saat ini.")
        else:
            for r in results:
                gid = int(r["gid"])
                with st.container(border=True):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if isinstance(r["place_img"], str) and r["place_img"].startswith(("http://", "https://")):
                            st.image(r["place_img"], width='stretch')
                    with cols[1]:
                        st.subheader(r["place_name"] or "-")
                        st.markdown(f"**Kategori:** {r['category'] or '-'}  \n**Kota:** {r['city'] or '-'}")
                        st.markdown(f"**Rating:** {'-' if r['rating'] is None else round(r['rating'], 2)}  \n**Harga:** {format_idr(r['price'])}")
                        if isinstance(r["place_map"], str) and r["place_map"].startswith(("http://", "https://")):
                            st.link_button("Buka peta", r["place_map"], width='content')
                        st.caption(f"Skor (setelah rerank): {r['score']:.4f}")

                        with st.expander("Lihat deskripsi"):
                            st.write(r.get("description") or "-")

                        # Actions
                        b1, b2, b3, _ = st.columns([1,1,1,6])
                        with b1:
                            if st.button("⭐ Suka", key=f"like_s_{gid}"):
                                like_item(gid); st.rerun()
                        with b2:
                            if st.button("🚫 Skip", key=f"skip_s_{gid}"):
                                skip_item(gid); st.rerun()
                        with b3:
                            label = "Hapus 🔖" if gid in st.session_state.bookmarked_idx else "🔖 Bookmark"
                            if st.button(label, key=f"bm_s_{gid}"):
                                toggle_bookmark(gid); st.rerun()
    else:
        st.info("Masukkan kueri lalu klik **Cari** untuk melihat hasil Search/KB.")

# ----------------------------- BOOKMARKS -----------------------------------
with tab_book:
    bms = list(st.session_state.bookmarked_idx)
    if not bms:
        st.info("Belum ada item yang di-bookmark.")
    else:
        st.success(f"{len(bms)} item di-bookmark.")
        for gid in bms:
            row = items.iloc[int(gid)]
            card = {
                "gid": int(gid),
                "place_name": row.get("place_name"),
                "place_img": row.get("place_img"),
                "place_map": row.get("place_map"),
                "category": row.get("category"),
                "city": row.get("city"),
                "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
                "price": None if pd.isna(row.get("price")) else float(row.get("price")),
                "description": get_description(row),
            }
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if isinstance(card["place_img"], str) and card["place_img"].startswith(("http://", "https://")):
                        st.image(card["place_img"], width='stretch')
                with cols[1]:
                    st.subheader(card["place_name"] or "-")
                    st.markdown(f"**Kategori:** {card['category'] or '-'}  \n**Kota:** {card['city'] or '-'}")
                    st.markdown(f"**Rating:** {'-' if card['rating'] is None else round(card['rating'], 2)}  \n**Harga:** {format_idr(card['price'])}")
                    if isinstance(card["place_map"], str) and card["place_map"].startswith(("http://", "https://")):
                        st.link_button("Buka peta", card["place_map"], width='content')
                    with st.expander("Lihat deskripsi"):
                        st.write(card.get("description") or "-")

                    # Actions
                    b1, b2, b3, _ = st.columns([1,1,1,6])
                    with b1:
                        if st.button("⭐ Suka", key=f"like_b_{gid}"):
                            like_item(gid); st.rerun()
                    with b2:
                        if st.button("🚫 Skip", key=f"skip_b_{gid}"):
                            skip_item(gid); st.rerun()
                    with b3:
                        if st.button("Hapus 🔖", key=f"bm_b_{gid}"):
                            toggle_bookmark(gid); st.rerun()

# Footer
st.write("---")
st.caption("CBF • Search/KB otomatis dari TF-IDF • Preferensi Like/Skip/Bookmark bersifat sementara di session.")
