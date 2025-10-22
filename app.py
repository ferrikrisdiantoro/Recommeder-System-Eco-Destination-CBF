# Streamlit CBF Recommender (EcoTourism) — no login
import streamlit as st
import pandas as pd
import numpy as np
import joblib, re, json, os, random
from pathlib import Path
from scipy.sparse import load_npz, csr_matrix
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="EcoTourism CBF Recommender", page_icon="🌿", layout="wide")

BASE_DIR = Path(__file__).parent
ART_DIR = BASE_DIR / "artifacts"
random.seed(13)

STOPWORDS_ID = set(["ada", "adalah", "agar", "akan", "antara", "atau", "banyak", "beberapa", "belum", "berbagai", "bila", "bisa", "bukan", "dalam", "dan", "dapat", "dari", "dengan", "di", "hanya", "harus", "hingga", "ini", "itu", "jika", "juga", "kah", "kami", "kamu", "karena", "ke", "kemudian", "kepada", "lah", "lain", "lainnya", "lalu", "lebih", "masih", "mereka", "mungkin", "namun", "nya", "oleh", "pada", "para", "pernah", "pun", "saat", "saja", "sampai", "sangat", "sebagai", "sebuah", "seluruh", "semua", "serta", "setiap", "suatu", "sudah", "supaya", "tanpa", "tapi", "tentang", "tentu", "terhadap", "tiap", "untuk", "yaitu", "yakni", "yang"])
IMPORTANT_WORDS = set(["di", "ke", "dari", "untuk", "dengan", "yang"])  # kata depan penting jangan dihapus

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    filtered = [t for t in tokens if (t not in STOPWORDS_ID) or (t in IMPORTANT_WORDS)]
    return " ".join(filtered)

@st.cache_resource(show_spinner=True)
def load_artifacts():
    items = pd.read_csv(ART_DIR / "items.csv")
    vec = joblib.load(ART_DIR / "vectorizer.joblib")
    X = load_npz(ART_DIR / "tfidf_matrix.npz").tocsr()
    # Try to load prebuilt neighbors, else build on the fly
    nbrs_path = ART_DIR / "nbrs_cosine.joblib"
    if nbrs_path.exists():
        nbrs = joblib.load(nbrs_path)
    else:
        nbrs = NearestNeighbors(n_neighbors=50, metric="cosine", algorithm="brute")
        nbrs.fit(X)
    return items, vec, X, nbrs

def format_idr(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "-"
    try:
        x = float(x)
        return "Rp{:,.0f}".format(x).replace(",", ".")
    except Exception:
        return str(x)

def recommend(input_text, items, vectorizer, X, top_n=10, threshold=0.3,
              categories=None, cities=None, max_price=None,
              diversify_per_category=3, serendipity_pct=20, keywords=None):
    if not input_text and not keywords:
        return []
    if keywords is None: keywords = []
    processed_input = preprocess_text(input_text)
    kw_all = [processed_input] + [preprocess_text(k) for k in keywords]

    mask = np.full(len(items), True, dtype=bool)
    if categories:
        cat_mask = items["category"].fillna("").apply(lambda s: any(c.lower() in s.lower() for c in categories)).values
        mask &= cat_mask
    if cities:
        city_mask = items["city"].fillna("").apply(lambda s: s.lower() in [c.lower() for c in cities]).values
        mask &= city_mask
    if max_price is not None:
        p = items["price"].fillna(np.inf).values.astype(float)
        mask &= p <= float(max_price)

    idx_all = np.arange(len(items))[mask]
    if idx_all.size == 0: return []

    def kw_count(s):
        s = str(s)
        return sum(1 for kw in kw_all if kw and kw in s)
    kcounts = items.loc[idx_all, "gabungan"].apply(kw_count).values

    Xsub = X[idx_all]
    sub_nbrs = NearestNeighbors(n_neighbors=min(top_n*4, Xsub.shape[0]), metric="cosine", algorithm="brute")
    sub_nbrs.fit(Xsub)

    q = vectorizer.transform([processed_input])
    dists, inds = sub_nbrs.kneighbors(q)

    cand = []
    seen = {}
    for r in range(inds.shape[1]):
        loc = inds[0, r]
        gid = idx_all[loc]
        sim = 1.0 - float(dists[0, r])
        if sim < threshold: continue
        row = items.iloc[gid]
        cat = str(row["category"]).split(",")[0].strip()
        if diversify_per_category and seen.get(cat, 0) >= diversify_per_category:
            continue
        seen[cat] = seen.get(cat, 0) + 1
        score = sim + 0.2 * float(kcounts[loc])
        cand.append((gid, score))

    if len(cand) < top_n and threshold > 0.0:
        for r in range(inds.shape[1]):
            loc = inds[0, r]
            gid = idx_all[loc]
            sim = 1.0 - float(dists[0, r])
            if sim < max(0.0, threshold-0.1): continue
            row = items.iloc[gid]
            cat = str(row["category"]).split(",")[0].strip()
            if diversify_per_category and seen.get(cat, 0) >= diversify_per_category:
                continue
            seen[cat] = seen.get(cat, 0) + 1
            score = sim + 0.2 * float(kcounts[loc])
            cand.append((gid, score))

    uniq = {}
    for i, sc in cand:
        if i not in uniq or sc > uniq[i]:
            uniq[i] = sc

    selected = set(uniq.keys())
    ser_pct = max(0, min(int(serendipity_pct), 30))
    n_ser = max(0, min(max(1, top_n//5), int(len(idx_all)*ser_pct/100)))
    if n_ser > 0:
        pool = [i for i in idx_all if i not in selected]
        if pool:
            pop_sorted = list(items.iloc[pool].sort_values("rating", ascending=False).index)
            random.shuffle(pop_sorted)
            for i in pop_sorted[:n_ser]:
                uniq.setdefault(i, 0.0)

    pairs = sorted(uniq.items(), key=lambda x:x[1], reverse=True)[:top_n]
    out = []
    for i, sc in pairs:
        row = items.iloc[int(i)]
        out.append({
            "place_name": row["place_name"],
            "place_img": row["place_img"],
            "place_map": row["place_map"],
            "category": row["category"],
            "city": row["city"],
            "rating": None if pd.isna(row["rating"]) else float(row["rating"]),
            "price": None if pd.isna(row["price"]) else float(row["price"]),
            "combined_score": float(sc),
        })
    return out

st.title("🌿 EcoTourism Recommender — CBF")

try:
    items, vectorizer, X, nbrs = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artifacts: {e}")
    st.stop()

st.sidebar.header("Filter")
all_categories = sorted(set(sum([str(c).split(",") for c in items["category"].fillna("").tolist()], [])))
all_categories = [c.strip() for c in all_categories if c.strip()]
sel_cats = st.sidebar.multiselect("Kategori", options=all_categories)

all_cities = sorted([c for c in items["city"].fillna("").unique() if c])
sel_cities = st.sidebar.multiselect("Kota/Kabupaten", options=all_cities)

min_price = float(np.nanmin(items["price"].values)) if items["price"].notna().any() else 0.0
max_price = float(np.nanmax(items["price"].values)) if items["price"].notna().any() else 0.0
use_price = st.sidebar.checkbox("Batasi harga maksimum", value=False)
if use_price:
    price_cap = st.sidebar.slider("Harga Maksimum (IDR)", min_value=0.0, max_value=max_price, value=min(max_price, 100000.0), step=1000.0)
else:
    price_cap = None

st.sidebar.header("Advanced")
top_n = st.sidebar.slider("Jumlah rekomendasi (Top‑N)", 5, 30, 10, 1)
threshold = st.sidebar.slider("Ambang kemiripan (cosine)", 0.0, 0.9, 0.3, 0.05)
div_per_cat = st.sidebar.slider("Diversifikasi per kategori", 0, 5, 3, 1)
serendip = st.sidebar.slider("Serendipity (%)", 0, 30, 20, 5)

query = st.text_input("Apa yang ingin kamu cari?", placeholder="contoh: pantai aceh, wisata alam bengkulu, hiking, snorkeling, dll")
kw_raw = st.text_input("Kata kunci tambahan (opsional, pisahkan dengan koma)", placeholder="pantai, laut, pasir putih")

col_go, col_info = st.columns([1,3])
with col_go:
    do_search = st.button("🔎 Cari Rekomendasi", use_container_width=True)
with col_info:
    st.write("")

if do_search:
    keywords = [k.strip() for k in kw_raw.split(",") if k.strip()] if kw_raw else []
    res = recommend(
        query, items, vectorizer, X, top_n=top_n, threshold=threshold,
        categories=sel_cats or None, cities=sel_cities or None, max_price=price_cap,
        diversify_per_category=div_per_cat, serendipity_pct=serendip, keywords=keywords
    )
    if not res:
        st.warning("Tidak ada hasil yang cocok. Coba longgarkan filter atau turunkan threshold.")
    else:
        for r in res:
            with st.container(border=True):
                cols = st.columns([1,3])
                with cols[0]:
                    if isinstance(r["place_img"], str) and r["place_img"].startswith(("http://","https://")):
                        st.image(r["place_img"], use_container_width=True)
                with cols[1]:
                    st.subheader(r["place_name"] or "-")
                    st.markdown(f"**Kategori:** {r['category'] or '-'}  \n**Kota:** {r['city'] or '-'}")
                    st.markdown(f"**Rating:** {'-' if r['rating'] is None else round(r['rating'],2)}  \n**Harga:** {format_idr(r['price'])}")
                    if isinstance(r["place_map"], str) and r["place_map"].startswith(("http://","https://")):
                        st.link_button("Buka peta", r["place_map"], use_container_width=False)
                    st.caption(f"Skor : {r['combined_score']:.4f}")
else:
    st.info("Masukkan kueri dan tekan **Cari Rekomendasi** untuk memulai.")
