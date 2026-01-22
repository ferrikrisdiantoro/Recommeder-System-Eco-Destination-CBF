"""
ufw.py - User Feedback Weighting (UFW) Module
==============================================
Modul ini berisi algoritma untuk menyesuaikan ranking rekomendasi 
berdasarkan feedback (Like/Skip) dari user selama sesi.

Konsep Utama:
- UFW menggunakan preferensi user untuk mempersonalisasi hasil
- Item yang mirip dengan item yang di-Like akan mendapat boost skor
- Item yang di-Skip akan mendapat penalty
- Kategori yang sering di-Like juga mendapat bonus

Analogi Sederhana:
    Bayangkan UFW seperti pelayan restoran yang pintar:
    - Jika kamu suka menu A dan B (Like), pelayan akan menawarkan 
      menu lain yang mirip dengan A dan B
    - Jika kamu bilang tidak mau menu C (Skip), pelayan tidak akan 
      menawarkan menu C lagi
    - Jika kamu suka 3 menu seafood, pelayan akan lebih banyak 
      menawarkan menu seafood lainnya

Rumus UFW:
    skor_akhir = skor_dasar 
                 + α × similarity_ke_centroid_like
                 - β × (1 jika item di-skip, 0 jika tidak)
                 + γ × jumlah_like_pada_kategori_ini
    
    Dimana:
    - α (alpha): Seberapa besar pengaruh "mirip dengan item yang di-Like"
    - β (beta): Seberapa besar penalty untuk item yang di-Skip
    - γ (gamma): Seberapa besar bonus untuk kategori favorit
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from .utils import normalize_minmax


# =============================================================================
# HELPER FUNCTIONS - Fungsi-fungsi pembantu yang lebih kecil
# =============================================================================

def _compute_centroid_dense(X, indices: List[int]) -> Optional[np.ndarray]:
    """
    Menghitung centroid (titik pusat) dari item-item yang di-Like.
    
    Apa itu Centroid?
        Centroid adalah "rata-rata" dari semua vektor item yang di-Like.
        Bayangkan kamu punya 3 titik di peta, centroid adalah titik tengahnya.
    
    Analogi:
        Jika kamu Like 3 pantai: Pantai A, Pantai B, Pantai C,
        centroid adalah "representasi ideal" dari pantai favoritmu.
        Item baru yang mirip dengan centroid ini kemungkinan kamu suka juga.
    
    Args:
        X: Matriks TF-IDF (semua item)
        indices: List index item yang di-Like
    
    Returns:
        np.ndarray: Vektor centroid (1 x n_features), atau None jika kosong
    
    Contoh:
        >>> liked_items = [5, 12, 23]  # User sudah Like item 5, 12, 23
        >>> centroid = _compute_centroid_dense(X, liked_items)
        >>> # centroid sekarang mewakili "preferensi rata-rata" user
    """
    # Jika tidak ada item yang di-Like, return None
    if not indices:
        return None
    
    # Ambil vektor TF-IDF dari semua item yang di-Like
    liked_vectors = X[list(indices)]
    
    # Hitung rata-rata (mean) dari semua vektor
    mean_vector = liked_vectors.mean(axis=0)
    
    # Konversi ke numpy array 2D (1 x n_features) untuk cosine_similarity
    # Handling sparse matrix dengan getattr untuk "A" attribute
    centroid = np.asarray(getattr(mean_vector, "A", mean_vector)).reshape(1, -1)
    
    return centroid


def _build_category_preference(
    items: pd.DataFrame, 
    liked_indices: List[int]
) -> Dict[str, int]:
    """
    Membangun dictionary preferensi kategori dari item yang di-Like.
    
    Cara Kerja:
        - Lihat semua item yang di-Like
        - Hitung berapa kali setiap kategori muncul
        - Return sebagai dictionary {kategori: jumlah}
    
    Contoh:
        Jika user Like 3 item dengan kategori:
        - Item 1: "Pantai"
        - Item 2: "Pantai" 
        - Item 3: "Gunung"
        
        Maka hasilnya: {"Pantai": 2, "Gunung": 1}
        
        Artinya user lebih suka Pantai, jadi item Pantai lain
        akan mendapat bonus skor lebih besar.
    
    Args:
        items: DataFrame item
        liked_indices: List index item yang di-Like
    
    Returns:
        Dict[str, int]: Dictionary {nama_kategori: jumlah_like}
    """
    category_preference: Dict[str, int] = {}
    
    if not liked_indices:
        return category_preference
    
    # Ambil kategori dari semua item yang di-Like
    liked_categories = (
        items.iloc[liked_indices]["category"]
        .fillna("")  # Handle NaN
        .apply(lambda s: str(s).split(",")[0].strip())  # Ambil kategori pertama
    )
    
    # Hitung frekuensi setiap kategori
    for category in liked_categories:
        if category:  # Skip string kosong
            category_preference[category] = category_preference.get(category, 0) + 1
    
    return category_preference


def _compute_like_similarity(
    gids: List[int], 
    X, 
    centroid: Optional[np.ndarray]
) -> np.ndarray:
    """
    Menghitung similarity setiap item ke centroid Like.
    
    Cara Kerja:
        1. Jika ada centroid (user sudah Like beberapa item):
           - Hitung cosine similarity setiap kandidat ke centroid
           - Normalisasi ke range 0-1
        2. Jika tidak ada centroid:
           - Return array 0 (tidak ada boost)
    
    Args:
        gids: List index item kandidat
        X: Matriks TF-IDF
        centroid: Vektor centroid dari item yang di-Like (atau None)
    
    Returns:
        np.ndarray: Array similarity scores (0-1) untuk setiap kandidat
    
    Contoh:
        >>> similarities = _compute_like_similarity([1, 2, 3], X, centroid)
        >>> print(similarities)  # [0.8, 0.3, 0.6]
        >>> # Item 1 paling mirip dengan preferensi user (0.8)
    """
    # Jika tidak ada centroid atau tidak ada kandidat
    if centroid is None or len(gids) == 0:
        return np.zeros(len(gids), dtype=float)
    
    # Hitung cosine similarity antara setiap kandidat dan centroid
    similarities = cosine_similarity(X[gids], centroid)[:, 0]
    
    # Normalisasi ke range 0-1 untuk konsistensi
    normalized_similarities = normalize_minmax(similarities)
    
    return normalized_similarities


def _calculate_final_scores(
    gids: List[int],
    base_scores_map: Dict[int, float],
    like_similarities: np.ndarray,
    blocked_gids: Set[int],
    category_preference: Dict[str, int],
    items: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    has_likes: bool
) -> Dict[int, float]:
    """
    Menghitung skor akhir UFW untuk setiap item.
    
    Rumus:
        skor_akhir = skor_dasar 
                     + α × similarity_like (jika ada Like)
                     - β × 1 (jika item di-Skip)
                     + γ × jumlah_like_kategori
    
    Penjelasan Parameter:
        - α (alpha): Boost untuk item yang mirip dengan item yang di-Like
          Nilai tinggi = lebih personalisasi berdasarkan Like
          
        - β (beta): Penalty untuk item yang di-Skip
          Nilai tinggi = item yang di-Skip sangat dihindari
          
        - γ (gamma): Boost untuk kategori favorit
          Nilai tinggi = kategori yang sering di-Like sangat diprioritaskan
    
    Args:
        gids: List index item kandidat
        base_scores_map: Dictionary {gid: skor_dasar}
        like_similarities: Array similarity ke centroid Like
        blocked_gids: Set index item yang di-Skip
        category_preference: Dictionary preferensi kategori
        items: DataFrame items
        alpha, beta, gamma: Parameter UFW
        has_likes: Boolean apakah user sudah Like item
    
    Returns:
        Dict[int, float]: Dictionary {gid: skor_akhir}
    """
    final_scores: Dict[int, float] = {}
    
    for i, gid in enumerate(gids):
        # Mulai dari skor dasar
        score = float(base_scores_map.get(gid, 0.0))
        
        # Komponen 1: Boost dari similarity ke Like centroid
        # Hanya berlaku jika user sudah Like minimal 1 item
        if has_likes:
            score += alpha * like_similarities[i]
        
        # Komponen 2: Penalty jika item di-Skip
        # Item yang di-Skip akan turun skornya
        if gid in blocked_gids:
            score -= beta
        
        # Komponen 3: Boost dari preferensi kategori
        # Kategori yang sering di-Like mendapat bonus
        category = str(items.iloc[gid]["category"]).split(",")[0].strip()
        if category and category_preference:
            like_count = category_preference.get(category, 0)
            score += gamma * like_count
        
        final_scores[gid] = score
    
    return final_scores


# =============================================================================
# MAIN FUNCTION - Fungsi utama UFW
# =============================================================================

def apply_ufw(
    gids: List[int], 
    base_scores_map: Dict[int, float], 
    X, 
    items: pd.DataFrame,
    alpha: float = 0.6, 
    beta: float = 0.7, 
    gamma: float = 0.02
) -> List[Tuple[int, float]]:
    """
    Menerapkan User Feedback Weighting (UFW) untuk re-ranking hasil.
    
    Ini adalah fungsi UTAMA untuk personalisasi rekomendasi berdasarkan
    feedback user selama sesi (Like/Skip).
    
    Alur Kerja:
        1. Ambil data Like dan Skip dari session_state
        2. Bangun preferensi kategori dari item yang di-Like
        3. Hitung centroid (titik pusat) dari item yang di-Like
        4. Hitung similarity setiap kandidat ke centroid
        5. Hitung skor akhir dengan rumus UFW
        6. Urutkan hasil berdasarkan skor akhir (descending)
    
    Analogi Lengkap:
        Bayangkan kamu di toko buku:
        
        1. Kamu ambil 3 buku dan bilang "saya suka ini" (Like)
           → Sistem mencatat: "Oh, user suka novel misteri"
           
        2. Kamu lihat 1 buku dan bilang "tidak tertarik" (Skip)
           → Sistem mencatat: "Jangan tampilkan buku ini lagi"
           
        3. Sistem kemudian:
           - Cari buku lain yang MIRIP dengan 3 buku yang kamu suka
           - Turunkan ranking buku yang kamu skip
           - Naikkan ranking buku dengan kategori yang sama
    
    Args:
        gids: List index item kandidat yang akan di-rerank
        base_scores_map: Dictionary {gid: skor_dasar} dari CBF
        X: Matriks TF-IDF untuk menghitung similarity
        items: DataFrame metadata item
        alpha: Boost weight untuk similarity ke Like (default: 0.6)
            - 0.0 = tidak ada pengaruh dari Like
            - 1.0 = pengaruh besar dari Like
        beta: Penalty weight untuk item yang di-Skip (default: 0.7)
            - 0.0 = tidak ada penalty untuk Skip
            - 1.0 = penalty besar untuk Skip
        gamma: Boost weight untuk kategori favorit (default: 0.02)
            - 0.0 = tidak ada bonus kategori
            - 0.1 = bonus signifikan untuk kategori favorit
    
    Returns:
        List[(gid, score)]: Daftar tuple (index item, skor akhir),
                           diurutkan dari skor tertinggi ke terendah
    
    Contoh Penggunaan:
        >>> # Dari hasil CBF
        >>> candidates = [(1, 0.8), (2, 0.7), (3, 0.9)]
        >>> gids = [1, 2, 3]
        >>> base_scores = {1: 0.8, 2: 0.7, 3: 0.9}
        >>> 
        >>> # Apply UFW
        >>> reranked = apply_ufw(gids, base_scores, X, items)
        >>> 
        >>> # Hasil mungkin berbeda dari CBF karena personalisasi
        >>> print(reranked)  # [(3, 1.2), (1, 0.9), (2, 0.3)]
    
    Contoh Dampak UFW:
        Sebelum UFW (dari CBF):
            1. Pantai A (skor: 0.9)
            2. Gunung B (skor: 0.85)
            3. Pantai C (skor: 0.8)
            
        User Like: Pantai A, Pantai C
        User Skip: Gunung B
        
        Setelah UFW:
            1. Pantai C (skor: 1.3) ← naik karena mirip dengan Like
            2. Pantai A (skor: 1.2) ← tetap tinggi
            3. Gunung B (skor: 0.15) ← turun drastis karena di-Skip
    """
    # Step 1: Ambil data feedback dari session_state Streamlit
    liked_indices: List[int] = list(st.session_state.liked_idx)
    blocked_indices: Set[int] = set(st.session_state.blocked_idx)
    
    # Step 2: Bangun preferensi kategori dari item yang di-Like
    category_preference = _build_category_preference(items, liked_indices)
    
    # Step 3: Hitung centroid dari item yang di-Like
    centroid = _compute_centroid_dense(X, liked_indices)
    
    # Step 4: Hitung similarity setiap kandidat ke centroid
    like_similarities = _compute_like_similarity(gids, X, centroid)
    
    # Step 5: Hitung skor akhir UFW
    final_scores = _calculate_final_scores(
        gids=gids,
        base_scores_map=base_scores_map,
        like_similarities=like_similarities,
        blocked_gids=blocked_indices,
        category_preference=category_preference,
        items=items,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        has_likes=len(liked_indices) > 0
    )
    
    # Step 6: Urutkan berdasarkan skor akhir (tertinggi dulu)
    sorted_results = sorted(
        final_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return sorted_results
