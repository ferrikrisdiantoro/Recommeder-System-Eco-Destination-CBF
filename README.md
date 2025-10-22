# EcoTourism CBF Recommender

Sistem rekomendasi **Content‑Based Filtering (CBF)** untuk destinasi ekowisata Indonesia. Pipeline terdiri dari *notebook* untuk menghasilkan artefak model dan aplikasi **Streamlit** untuk inferensi.

## Struktur Projek
```
├─ artifacts/                  # model/cbf
│  ├─ items.csv
│  ├─ tfidf_matrix.npz
│  ├─ vectorizer.joblib
│  ├─ nbrs_cosine.joblib       # opsional
│  └─ metadata.json
├─ dataset/
│  └─ eco_place.csv         # data destinasi wisata
├─ docs/
│  ├─ `System Diagram.drawio`       # opsional
│  └─ `System Diagram.png`
├─ notebook/                    
│  └─ indo_ecotourism_cbf.ipynb
├─ app.py                      # aplikasi Streamlit
└─ requirements.txt
```

## Alur Sistem
1. **Notebook** membersihkan data, membangun fitur **TF‑IDF**, melatih indeks **NearestNeighbors (cosine)** dan menyimpan artefak ke `artifacts/`.
2. **Streamlit** memuat artefak, menerima kueri + filter (kategori, kota, harga), lalu menghitung kemiripan **cosine** untuk menghasilkan rekomendasi dengan **diversifikasi kategori** dan **serendipity**.

Lihat diagram PlantUML di `docs/system_flow.puml`.

## Menjalankan Lokal
```bash
# 1) (opsional) buat venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) pastikan folder artifacts/ berisi files yang diperlukan

# 4) run app
streamlit run app.py
```

## Deploy ke Streamlit Cloud
1. Push repo ini ke **GitHub**.
2. Buka https://share.streamlit.io dan pilih repo + branch + path `app.py`.
3. Dependensi akan diinstal dari `requirements.txt`.
4. Pastikan folder `artifacts/` ikut ter-*commit*.

## Dataset
- `eco_place.csv` dari dataset publik *Indonesia EcoTourism*.

## Catatan Desain
- *CBF only*.
- Preprocessing ringan (**stopwords** Indonesia built‑in, *tanpa stemming* untuk menjaga konteks).
- **TF‑IDF (L2)** + **cosine similarity**.
- **Diversifikasi kategori** (maks N per kategori) untuk mengurangi overspecialization.
- **Serendipity** (0–30%) menyuntik beberapa item populer di luar daftar paling mirip.
