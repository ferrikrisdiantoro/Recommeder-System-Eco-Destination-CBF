import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
from scipy import sparse
from pathlib import Path
import os
import sys
import re

# Add current dir to path to find local modules
sys.path.append(os.getcwd())

# Coba import preprocess_text, kalau gagal pake fallback sederhana
try:
    from eco_recsys.text import preprocess_text
    print("Algorithm: Menggunakan simple preprocessing dari eco_recsys...")
except ImportError:
    print("Algorithm: Menggunakan fallback preprocessing...")
    def preprocess_text(text: str) -> str:
        text = str(text).lower()
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        return " ".join(tokens)

def rebuild():
    print("\n=== REBUILDING ARTIFACTS (Local) ===")
    print("Tujuan: Mengatasi 'NotFittedError' atau perbedaan versi scikit-learn.\n")
    
    base_dir = Path(os.getcwd())
    data_path = base_dir / "dataset" / "eco_place.csv"
    art_dir = base_dir / "artifacts"
    
    # Buat folder artifacts jika belum ada
    art_dir.mkdir(exist_ok=True)
    
    # 1. Load Dataset
    if not data_path.exists():
        print(f"ERROR: Dataset tidak ditemukan di {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # 2. Preprocessing Columns
    # Kita perlu membuat kolom 'gabungan' untuk TF-IDF
    # Gabungan = Category + Place_Name + City + Description
    print("Creating 'gabungan' column...")
    
    def make_soup(row):
        cols = [
            str(row.get('category', '')),
            str(row.get('place_name', '')),
            str(row.get('city', '')),
            str(row.get('place_description', ''))
        ]
        return " ".join(cols)

    df['gabungan'] = df.apply(make_soup, axis=1)
    
    # Preprocess text (cleansing)
    print("Preprocessing text (might take a moment)...")
    df['gabungan'] = df['gabungan'].apply(preprocess_text)

    # --- CLEANING DATA (Price & Rating) ---
    print("Cleaning Price and Rating columns...")
    def clean_price(val):
        # Convert 'Rp900,000' -> 900000.0
        val_str = str(val)
        if val_str.lower() == 'nan': return 0.0
        # Hapus Rp, koma, spasi
        clean = val_str.replace("Rp", "").replace("rp", "").replace(",", "").replace(" ", "").strip()
        try:
            return float(clean)
        except ValueError:
            return 0.0

    df['price'] = df['price'].apply(clean_price)
    
    # Pastikan rating juga float
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    
    # Save processed items to items.csv
    items_path = art_dir / "items.csv"
    print(f"Saving items to {items_path}...")
    df.to_csv(items_path, index=False)
    
    # 3. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    # Gunakan setting standar
    vectorizer = TfidfVectorizer(max_features=5000) 
    tfidf_matrix = vectorizer.fit_transform(df['gabungan'])
    
    print(f"TF-IDF Shape: {tfidf_matrix.shape}")
    
    # Save Vectorizer & Matrix
    print("Saving vectorizer & matrix...")
    joblib.dump(vectorizer, art_dir / "vectorizer.joblib")
    sparse.save_npz(art_dir / "tfidf_matrix.npz", tfidf_matrix)
    
    # 4. NearestNeighbors Model
    print("Fitting NearestNeighbors model...")
    nbrs = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    nbrs.fit(tfidf_matrix)
    
    print("Saving NearestNeighbors model...")
    joblib.dump(nbrs, art_dir / "nbrs_cosine.joblib")
    
    print("\n=== SUCCESS! ARTIFACTS REBUILT ===")
    print("[OK] Semua model sudah diperbarui sesuai versi Python/scikit-learn di komputer ini.")
    print(">> Silakan RESTART aplikasi Streamlit Anda sekarang.")

if __name__ == "__main__":
    rebuild()
