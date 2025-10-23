import streamlit as st
import pandas as pd
import numpy as np
import joblib, random, math
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="EcoTourism CBF", page_icon="🌿", layout="wide")
random.seed(13)

BASE_DIR = Path(__file__).parent
ART_DIR  = BASE_DIR / "artifacts"

@st.cache_resource(show_spinner=True)
def load_artifacts():
    """
    Memuat artefak yang dibangun dari notebook:
      - items.csv        : metadata + kolom 'gabungan' (teks terproses)
      - tfidf_matrix.npz : matriks fitur item (CSR sparse)
      - vectorizer.joblib: (opsional di app ini; tidak dipakai untuk feed tanpa query)
    """
    items = pd.read_csv(ART_DIR / "items.csv")
    # Vectorizer dimuat jika suatu saat perlu KB/pencarian; tidak dipakai untuk feed awal.
    try:
        vec = joblib.load(ART_DIR / "vectorizer.joblib")
    except Exception:
        vec = None

    X = load_npz(ART_DIR / "tfidf_matrix.npz").tocsr()

    # Pastikan kolom-kolom UI minimal tersedia
    for col in ["place_name", "category", "city", "price", "rating", "place_img", "place_map", "gabungan"]:
        if col not in items.columns:
            items[col] = np.nan

    return items, vec, X

def format_idr(x):
    """Format angka ke Rupiah"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return "Rp{:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return str(x)

def init_session():
    # Simpan indeks global item yang di-Like/Skip selama sesi (tidak dipersist ke DB)
    st.session_state.setdefault("liked_idx", set())
    st.session_state.setdefault("blocked_idx", set())
init_session()

def like_item(gid: int):
    st.session_state.liked_idx.add(int(gid))
    st.session_state.blocked_idx.discard(int(gid))

def skip_item(gid: int):
    st.session_state.blocked_idx.add(int(gid))
    st.session_state.liked_idx.discard(int(gid))

def clear_feedback():
    st.session_state.liked_idx.clear()
    st.session_state.blocked_idx.clear()

# ========== Util numerik ==========
def normalize_minmax(a: np.ndarray) -> np.ndarray:
    """Normalisasi min-max; jika rentang nol -> vektor nol."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

def compute_centroid_dense(X, indices):
    """
    Hitung centroid (mean) dari baris-baris X[indices].
    Penting: scipy.sparse .mean() mengembalikan numpy.matrix -> konversi ke ndarray (1, d).
    """
    if not indices:
        return None
    rows = X[list(indices)]                   # CSR (k, d)
    mean_mat = rows.mean(axis=0)              # numpy.matrix (1, d)
    # Konversi aman ke ndarray 2D shape (1, d)
    if hasattr(mean_mat, "A"):
        cent = np.asarray(mean_mat.A).reshape(1, -1)
    else:
        cent = np.asarray(mean_mat).reshape(1, -1)
    return cent

def mmr_select(idx_all, X, base_scores_aligned, top_n=20, lambda_mmr=0.7,
               per_category_cap=0, items: pd.DataFrame | None = None):
    """
    Pilih set kandidat beragam via MMR:
      argmax  λ * base_score  −  (1 − λ) * max_sim_ke_item_terpilih
    - idx_all            : array indeks global (yang lolos filter)
    - base_scores_aligned: skor dasar per idx_all (mis. rating ter-normalisasi)
    - per_category_cap   : batasi jumlah per kategori utama (0 = tanpa batas)
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

            score = float(base_scores_aligned[loc])
            if not selected_loc:
                mmr = score
            else:
                sel_g = idx_all[selected_loc]                    # (m,)
                # cosine_similarity untuk (1, d) vs (m, d)
                sim = cosine_similarity(X[gid], X[sel_g]).max()  # ndarray -> float
                mmr = lambda_mmr * score - (1.0 - lambda_mmr) * float(sim)

            if mmr > best_val:
                best_val, best_loc = mmr, loc

        if best_loc is None:
            break

        selected_loc.append(best_loc)

        # Update hitung kategori
        if per_category_cap and items is not None:
            cat = str(items.iloc[int(idx_all[best_loc])]["category"]).split(",")[0].strip()
            if cat:
                cat_count[cat] = cat_count.get(cat, 0) + 1

        candidates.remove(best_loc)
        if len(selected_loc) >= top_n:
            break

    # Hasil dalam indeks global
    return [int(idx_all[loc]) for loc in selected_loc]

def rerank_with_feedback(gids, base_scores_map, X, items,
                         alpha=0.6, beta=0.7, gamma=0.02):
    """
    Skor akhir = base_score
                 + alpha * sim( item , centroid_like )
                 - beta  * 1[item di blocked]
                 + gamma * preferensi_kategori_dari_like
    """
    liked   = list(st.session_state.liked_idx)
    blocked = set(st.session_state.blocked_idx)

    # Preferensi kategori dari item yang di-Like
    cat_pref: dict[str, int] = {}
    if liked:
        liked_cats = items.iloc[liked]["category"].fillna("").apply(lambda s: str(s).split(",")[0].strip())
        for c in liked_cats:
            if c:
                cat_pref[c] = cat_pref.get(c, 0) + 1

    # Centroid Like -> ndarray (1, d)
    cent = compute_centroid_dense(X, liked)
    sims = np.zeros(len(gids), dtype=float)
    if cent is not None and len(gids):
        # cosine_similarity: (m, d) vs (1, d) -> (m, 1)
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

def build_feed(items, X, filters, top_n=12, mmr_lambda=0.7, per_cat_cap=2,
               serendipity_pct=15, use_feedback=True,
               alpha=0.6, beta=0.7, gamma=0.02):
    """
    1) Filter item sesuai sidebar
    2) Skor dasar = rating ternormalisasi (+ noise kecil untuk tie-break)
    3) Diversifikasi via MMR
    4) Sisipkan serendipity (top-rated di luar kandidat MMR)
    5) Rerank pakai Like/Skip (session)
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

    # 2) Skor dasar (popularity proxy: rating)
    rating = items.iloc[idx_all]["rating"].fillna(0.0).clip(lower=0.0).values.astype(float)
    base_scores_aligned = normalize_minmax(rating) + np.random.RandomState(13).rand(len(idx_all)) * 1e-4

    # 3) Diversifikasi
    gids = mmr_select(idx_all, X, base_scores_aligned, top_n=top_n,
                      lambda_mmr=mmr_lambda, per_category_cap=per_cat_cap, items=items)

    # Peta skor dasar untuk indeks global
    base_map = {int(idx_all[i]): float(base_scores_aligned[i]) for i in range(len(idx_all))}

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
            # Tandai serendipity dengan base score minimal
            for g in pick:
                base_map[int(g)] = min(base_map.get(int(g), 0.0), 0.0)

    # 5) Rerank dengan feedback sesi
    if use_feedback:
        pairs = rerank_with_feedback(gids, base_map, X, items, alpha=alpha, beta=beta, gamma=gamma)
    else:
        pairs = sorted([(g, base_map.get(g, 0.0)) for g in gids], key=lambda x: x[1], reverse=True)

    # 6) Kemasan output: deskripsi = place_description jika ada & berisi; fallback ke 'gabungan'
    out = []
    for gid, sc in pairs[:top_n]:
        row = items.iloc[int(gid)]
        desc_raw = row.get("place_description")
        if not (isinstance(desc_raw, str) and desc_raw.strip()):
            desc_raw = row.get("gabungan", "")
        out.append({
            "gid": int(gid),
            "place_name": row.get("place_name"),
            "place_img": row.get("place_img"),
            "place_map": row.get("place_map"),
            "category": row.get("category"),
            "city": row.get("city"),
            "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
            "price": None if pd.isna(row.get("price")) else float(row.get("price")),
            "description": desc_raw,
            "score": float(sc),
        })
    return out

st.title("🌿 EcoTourism Recommender — CBF Only (Auto Feed)")
st.caption("Feed otomatis tanpa query • Diversifikasi (MMR) • Serendipity • Reranking Like/Skip (session)")

# Muat artefak
try:
    items, vectorizer, X = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artifacts: {e}")
    st.stop()

# Sidebar: Filter & pengaturan
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
top_n       = st.sidebar.slider("Jumlah item (Top-N)", 5, 40, 12, 1)
mmr_lambda  = st.sidebar.slider("MMR λ (relevansi vs keragaman)", 0.0, 1.0, 0.7, 0.05)
per_cat_cap = st.sidebar.slider("Batas per kategori (0=tanpa batas)", 0, 6, 2, 1)
serendip    = st.sidebar.slider("Serendipity (%)", 0, 30, 15, 5)

st.sidebar.header("Session Feedback")
use_fb = st.sidebar.toggle("Aktifkan reranking Like/Skip", value=True)
alpha  = st.sidebar.slider("Boost ke Like (α)", 0.0, 2.0, 0.6, 0.05, disabled=not use_fb)
beta   = st.sidebar.slider("Penalty Skip (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
gamma  = st.sidebar.slider("Boost kategori Like (γ)", 0.0, 0.2, 0.02, 0.005, disabled=not use_fb)
if st.sidebar.button("Reset Like/Skip", type="secondary"):
    clear_feedback()
    st.sidebar.success("Preferensi sesi direset.")

# Status Like/Skip
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)} item")
    with c2:
        st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)} item")

# Bangun & render feed (auto, recompute setiap interaksi)
filters = {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}
feed = build_feed(
    items, X, filters,
    top_n=top_n, mmr_lambda=mmr_lambda, per_cat_cap=per_cat_cap,
    serendipity_pct=serendip, use_feedback=use_fb,
    alpha=alpha, beta=beta, gamma=gamma
)

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

                # Deskripsi: pakai place_description jika ada & berisi; fallback ke 'gabungan'
                desc = r.get("description") or ""
                with st.expander("Lihat deskripsi"):
                    st.write(desc if isinstance(desc, str) and desc.strip() else "-")

                # Aksi interaksi pengguna
                b1, b2, _ = st.columns([1, 1, 6])
                with b1:
                    if st.button("⭐ Suka", key=f"like_{gid}"):
                        like_item(gid)
                        st.rerun()   # Streamlit 1.30+: gunakan st.rerun()
                with b2:
                    if st.button("🚫 Skip", key=f"skip_{gid}"):
                        skip_item(gid)
                        st.rerun()

st.write("---")
st.caption("Mode: CBF. Interaksi Like/Skip mempengaruhi urutan rekomendasi pada sesi ini.")
