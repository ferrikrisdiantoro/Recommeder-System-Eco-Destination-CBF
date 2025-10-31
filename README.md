
# EcoTourism Recommender — **CBF + UFW**

Aplikasi rekomendasi destinasi wisata berbasis **Content‑Based Filtering (CBF)** dengan **User Feedback Weighting (UFW)** untuk personalisasi ringan.

---

## 1) Fitur Utama
- **CBF (TF‑IDF cosine)** untuk Feed & Search/KB.
- **UFW (User Feedback Weighting)**: reranking berbasis *centroid Like* + penalti Skip + bias kategori dari Like.
- **MMR Diversification** + **Serendipity** pada Feed.
- **Modular**: `eco_recsys.cbf`, `eco_recsys.ufw`, `eco_recsys.ui`, `eco_recsys.data`, `eco_recsys.state`, `eco_recsys.utils`, `eco_recsys.text`.
- **Session actions**: ⭐ Like, 🚫 Skip, 🔖 Bookmark (disimpan di `st.session_state`).

---

## 2) Artefak
Letakkan di folder `artifacts/` yang berdampingan dengan `app.py`:
- `items.csv` — metadata item (termasuk `gabungan`, `place_description`, `category`, `city`, `rating`, `price`, `place_img`, `place_map`).
- `tfidf_matrix.npz` — matriks fitur item (CSR).
- `vectorizer.joblib` — TF‑IDF vectorizer.
- `nbrs_cosine.joblib` —  indeks **NearestNeighbors(metric='cosine')** untuk percepat Search/KB.
- `metadata.json` — informasi ringkas.

> Artefak dihasilkan oleh notebook **`indo_ecotourism_cbf_ufw.ipynb`**.

---
## 3) Struktur Proyek
```
.
├─ app.py                        # Entrypoint Streamlit (ringkas)
├─ artifacts/                    # Artefak dari notebook
├  ├─ items.csv                  # Metadata item + kolom gabungan (teks terproses)
├  ├─ vectorizer.joblib          # TF-IDF vectorizer
├  ├─ tfidf_matrix.npz           # Matriks fitur item (CSR)
├  ├─ nbrs_cosine.joblib         # Indeks tetangga terdekat
├  └─ metadata.json              # Metadata 
├─ eco_recsys/                   # Package modular
├  ├─ __init__.py
├  ├─ data.py                    # Loader artifacts + kolom minimal
├  ├─ cbf.py                     # CBF feed & search + MMR
├  ├─ ufw.py                     # Reranking UFW (centroid Like)
├  ├─ ui.py                      # Komponen UI (sidebar, slider, card, tabs)
├  ├─ state.py                   # Session state: Like/Skip/Bookmark
├  ├─ utils.py                   # normalize_minmax, format_idr, get_description
├  └─ text.py                    # preprocess_text (stopwords, tokenisasi ringan)
├─ dataset/
├  └─ eco_place.csv              # Dataset proyek
├─ notebook/
└─ └─ indo_ecotourism_cbf_ufw.ipynb             # Kode pembuatan model CBF
```

---

## 4) Quickstart
### Instalasi
```bash
pip install streamlit scikit-learn scipy pandas numpy joblib
```
### Menjalankan
```bash
streamlit run app.py
```
Pastikan folder `artifacts/` berisi artefak dari notebook.

---

## 5) Cara Kerja (ringkas)

### 5.1 Feed (tanpa query) — `eco_recsys.cbf.build_feed_cbf(...)`
1. **Filter** (kategori/kota/harga) → subset indeks `idx_all`.
2. **Base score**: normalisasi `rating` (+ noise kecil untuk tie‑break).
3. **Diversifikasi** MMR: pilih kandidat `gids` dengan argmax  
   `λ·score − (1−λ)·max_sim_terhadap_yang_sudah_dipilih` (cosine pada TF‑IDF).
4. **Serendipity**: sisipkan sebagian item populer di luar kandidat utama.
5. **UFW**: hasil feed direrank via `eco_recsys.ufw.apply_ufw(...)`.

### 5.2 Search/KB (dengan query) — `eco_recsys.cbf.search_cbf(...)`
1. **Preprocess** query (lowercase, token, stopwords ringan dipertahankan kata depan penting).
2. Transform **TF‑IDF** via `vectorizer.transform([query])`.
3. **Jika** `nbrs_cosine.joblib` tersedia → gunakan `kneighbors` (`sim = 1 - dist`) sebagai kandidat cepat; **fallback** ke `cosine_similarity` penuh bila tidak ada.
4. **MMR** pada kandidat dengan skor kesamaan sebagai base.
5. **UFW** untuk reranking akhir.

### 5.3 UFW (User Feedback Weighting) — `eco_recsys.ufw.apply_ufw(...)`
- Bentuk **centroid Like** di ruang TF‑IDF dari item‑item yang di‑Like pada sesi.
- Hitung `sim(item, centroid_like)` (cosine) → **min–max normalize**.
- **Skor akhir** (per item):
  ```text
  score_final = base_score
              + α · sim_to_like_centroid
              − β · 1[item di blocked/Skip]
              + γ · preferensi_kategori_dari_Like
  ```

---

## 6) UI & Interaksi
- **Tabs**: 🏠 Feed, 🔎 Search/KB, 🔖 Bookmarks.
- **Sidebar**:
  - *Filter*: kategori, kota/kabupaten, harga maksimum.
  - *Feed knobs*: Top‑N, λ MMR, batas per kategori, serendipity %.
  - *UFW knobs*: toggle aktifkan reranking, α (Like), β (Skip), γ (kategori).
  - Tombol **Reset Like/Skip** untuk membersihkan preferensi sesi.
- **Card** item: gambar, nama, kategori, kota, rating, harga, tombol ke peta, deskripsi, skor (jika relevan).
- Aksi: ⭐ Like, 🚫 Skip, 🔖 Bookmark (memicu `st.rerun()` agar urutan terbarui).

---

## 7) Session State
Disimpan di `st.session_state` (per sesi pengguna di browser):
- `liked_idx: set[int]`
- `blocked_idx: set[int]`
- `bookmarked_idx: set[int]`

Helper ada di `eco_recsys.state`: `like_item`, `skip_item`, `toggle_bookmark`, `clear_feedback`.

---

## 8) Parameter Penting
- **MMR λ** (`0..1`): besar → lebih condong ke skor dasar (relevansi), kecil → lebih menekankan keragaman.
- **Top‑N** (Feed/Search).
- **α, β, γ** (UFW):
  - α: pengaruh kemiripan ke centroid Like.
  - β: penalti jika item di‑Skip.
  - γ: bias kategori dari Like.
- **Serendipity %**: fraksi tambahan item populer di luar kandidat utama.

---

## 9) Error Handling & Fallback
- **Vectorizer None** → tab Search menampilkan pesan agar menghasilkan artefak dari notebook.
- **URL gambar/peta** diverifikasi skema `http(s)://` sebelum dirender.
- **Kolom wajib** dipastikan ada via `ensure_min_columns` (nilai default `NaN` bila hilang).
- **NN index** (`nbrs_cosine.joblib`) opsional — otomatis fallback ke `cosine_similarity` penuh.

---

## 10) Diagram Alur (ringkas)
```
[User]
  ├─ set filters ───────► CBF Feed (rating→MMR→Serendipity) ─► (UFW) ─► Render
  ├─ type query ────────► TF-IDF (±NN index) ─► MMR ─► (UFW) ─► Render
  └─ Like/Skip/Bookmark ─► Update session_state ─► st.rerun() ─► ranking terbarui
```

---
