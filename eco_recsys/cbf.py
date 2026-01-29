"""
cbf.py - Content-Based Filtering (CBF) Module
==============================================
Modul ini berisi algoritma rekomendasi berbasis konten menggunakan TF-IDF dan cosine similarity.

Konsep Utama:
- TF-IDF (Term Frequency - Inverse Document Frequency): Teknik untuk mengubah teks menjadi angka
- Cosine Similarity: Mengukur kemiripan antar item berdasarkan sudut vektor
- MMR (Maximal Marginal Relevance): Menyeimbangkan relevansi dan keragaman hasil

Analogi Sederhana:
- CBF seperti sistem yang merekomendasikan restoran berdasarkan deskripsi menu
- Jika kamu suka "nasi goreng seafood", sistem akan mencari restoran dengan menu serupa
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError

from .text import preprocess_text
from .utils import normalize_minmax


# =============================================================================
# DATA CLASSES - Struktur data untuk menyimpan hasil
# =============================================================================

@dataclass
class Candidate:
    """
    Kelas untuk menyimpan kandidat item rekomendasi.
    
    Attributes:
        gid (int): Global ID / index item dalam DataFrame
        base_score (float): Skor dasar item (0.0 - 1.0)
    
    Contoh:
        candidate = Candidate(gid=42, base_score=0.85)
        # Artinya item ke-42 memiliki skor 0.85
    """
    gid: int
    base_score: float


# =============================================================================
# HELPER FUNCTIONS - Fungsi-fungsi pembantu yang lebih kecil
# =============================================================================

def _mask_by_filters(items: pd.DataFrame, filters: Dict) -> np.ndarray:
    """
    Membuat mask (boolean array) untuk memfilter item berdasarkan kriteria user.
    
    Analogi:
        Seperti "saringan" yang hanya meloloskan item yang sesuai kriteria.
        Jika user pilih kategori "Pantai" dan kota "Aceh", hanya item
        dengan kedua kriteria itu yang akan lolos (True).
    
    Args:
        items: DataFrame berisi semua item destinasi
        filters: Dictionary berisi kriteria filter:
            - categories (list): Daftar kategori yang dipilih
            - cities (list): Daftar kota yang dipilih  
            - max_price (float): Harga maksimum yang diinginkan
    
    Returns:
        np.ndarray: Array boolean, True = item lolos filter
    
    Contoh:
        filters = {"categories": ["Pantai"], "cities": ["Aceh"], "max_price": 50000}
        mask = _mask_by_filters(items, filters)
        # mask = [True, False, True, False, ...]
    """
    # Mulai dengan semua item True (lolos semua)
    mask = np.full(len(items), True, dtype=bool)
    
    # Filter 1: Kategori
    # Jika user memilih kategori tertentu, filter berdasarkan itu
    if filters.get("categories"):
        # Ubah semua kategori ke lowercase untuk perbandingan yang konsisten
        selected_cats = [c.lower() for c in filters["categories"]]
        
        # Cek apakah kategori item mengandung salah satu kategori yang dipilih
        mask &= items["category"].fillna("").apply(
            lambda s: any(cat in s.lower() for cat in selected_cats)
        ).values
    
    # Filter 2: Kota/Kabupaten
    if filters.get("cities"):
        selected_cities = [c.lower() for c in filters["cities"]]
        
        # Cek apakah kota item ada dalam daftar kota yang dipilih
        mask &= items["city"].fillna("").apply(
            lambda s: s.lower() in selected_cities
        ).values
    
    # Filter 3: Harga Maksimum
    if filters.get("max_price") is not None:
        # Ambil kolom harga, isi NaN dengan infinity agar tidak lolos filter
        prices = items["price"].fillna(np.inf).values.astype(float)
        
        # Hanya item dengan harga <= max_price yang lolos
        mask &= prices <= float(filters["max_price"])
    
    return mask


def _check_category_cap(
    item_gid: int, 
    items: pd.DataFrame, 
    cat_count: Dict[str, int], 
    per_category_cap: int
) -> bool:
    """
    Cek apakah kategori item sudah melebihi batas maksimum.
    
    Tujuan:
        Mencegah hasil didominasi oleh satu kategori saja.
        Misalnya, jika cap=2, maksimal 2 item "Pantai" yang bisa muncul.
    
    Args:
        item_gid: Index item yang akan dicek
        items: DataFrame berisi semua item
        cat_count: Dictionary counter kategori yang sudah dipilih
        per_category_cap: Batas maksimum per kategori
    
    Returns:
        bool: True jika kategori sudah penuh (skip item ini)
    """
    if per_category_cap <= 0:
        return False  # Tidak ada batasan
    
    # Ambil kategori pertama dari item (split by comma)
    category = str(items.iloc[item_gid]["category"]).split(",")[0].strip()
    
    if category and cat_count.get(category, 0) >= per_category_cap:
        return True  # Kategori sudah penuh
    
    return False


def _calculate_mmr_score(
    candidate_score: float,
    candidate_vector,
    selected_vectors,
    X,
    lambda_mmr: float
) -> float:
    """
    Menghitung skor MMR (Maximal Marginal Relevance) untuk satu kandidat.
    
    Rumus MMR:
        MMR = λ × (skor_relevansi) - (1-λ) × (similarity_dengan_item_terpilih)
    
    Analogi:
        Bayangkan kamu memilih film untuk marathon:
        - λ tinggi (0.9): Pilih film dengan rating tertinggi meski genre sama
        - λ rendah (0.3): Pilih film beragam meski rating lebih rendah
    
    Args:
        candidate_score: Skor dasar kandidat
        candidate_vector: Vektor TF-IDF kandidat
        selected_vectors: Vektor TF-IDF item yang sudah terpilih
        X: Matriks TF-IDF lengkap
        lambda_mmr: Parameter keseimbangan (0=diversitas, 1=relevansi)
    
    Returns:
        float: Skor MMR kandidat
    """
    if selected_vectors is None or len(selected_vectors) == 0:
        # Kandidat pertama, tidak ada yang dibandingkan
        return candidate_score
    
    # Hitung similarity dengan semua item yang sudah dipilih
    similarity_to_selected = cosine_similarity(candidate_vector, selected_vectors)
    max_similarity = float(similarity_to_selected.max())
    
    # Rumus MMR: tinggi jika relevan DAN berbeda dari yang sudah dipilih
    mmr_score = lambda_mmr * candidate_score - (1.0 - lambda_mmr) * max_similarity
    
    return mmr_score


def _update_category_count(
    item_gid: int, 
    items: pd.DataFrame, 
    cat_count: Dict[str, int]
) -> None:
    """
    Update counter kategori setelah item dipilih.
    
    Args:
        item_gid: Index item yang dipilih
        items: DataFrame items
        cat_count: Dictionary counter (akan dimodifikasi in-place)
    """
    category = str(items.iloc[item_gid]["category"]).split(",")[0].strip()
    if category:
        cat_count[category] = cat_count.get(category, 0) + 1


# =============================================================================
# CORE FUNCTIONS - Fungsi-fungsi utama algoritma CBF
# =============================================================================

def mmr_select(
    idx_all: np.ndarray, 
    X, 
    base_scores: np.ndarray, 
    top_n: int = 20, 
    lambda_mmr: float = 0.7,
    per_category_cap: int = 0, 
    items: Optional[pd.DataFrame] = None
) -> List[int]:
    """
    Memilih item menggunakan algoritma MMR (Maximal Marginal Relevance).
    
    Tujuan:
        Memilih item yang RELEVAN sekaligus BERAGAM.
        Menghindari hasil yang monoton (semua pantai, semua gunung, dll).
    
    Cara Kerja:
        1. Mulai dari kandidat dengan skor tertinggi
        2. Untuk setiap kandidat berikutnya:
           - Hitung skor MMR = relevansi - similarity dengan yang sudah dipilih
           - Pilih kandidat dengan skor MMR tertinggi
        3. Ulangi sampai jumlah yang diinginkan tercapai
    
    Args:
        idx_all: Array berisi index item yang menjadi kandidat
        X: Matriks TF-IDF (sparse matrix) untuk menghitung similarity
        base_scores: Array skor dasar untuk setiap kandidat
        top_n: Jumlah item yang ingin dipilih
        lambda_mmr: Parameter MMR (0.0-1.0)
            - 1.0 = prioritas relevansi (item mirip boleh)
            - 0.0 = prioritas diversitas (item harus beda)
            - 0.7 = default, seimbang
        per_category_cap: Batas maksimum item per kategori (0 = unlimited)
        items: DataFrame untuk mengecek kategori
    
    Returns:
        List[int]: Daftar index item yang terpilih (global ID)
    
    Contoh:
        >>> selected = mmr_select(idx_all, X, scores, top_n=10, lambda_mmr=0.7)
        >>> print(selected)  # [5, 12, 3, 27, ...]
    """
    selected_positions: List[int] = []  # Posisi dalam idx_all yang sudah dipilih
    remaining_candidates = list(range(len(idx_all)))  # Posisi yang masih tersedia
    category_count: Dict[str, int] = {}  # Counter per kategori
    
    # Batas maksimum kandidat yang dipertimbangkan (3x top_n untuk efisiensi)
    max_candidates = min(top_n * 3, len(idx_all))
    
    while remaining_candidates and len(selected_positions) < max_candidates:
        best_position = None
        best_mmr_score = -float('inf')  # Mulai dari nilai terendah
        
        # Evaluasi setiap kandidat yang tersisa
        for position in remaining_candidates:
            item_gid = int(idx_all[position])
            
            # Cek batas kategori jika diaktifkan
            if per_category_cap and items is not None:
                if _check_category_cap(item_gid, items, category_count, per_category_cap):
                    continue  # Skip, kategori sudah penuh
            
            # Hitung skor MMR
            candidate_base_score = float(base_scores[position])
            
            if not selected_positions:
                # Kandidat pertama, gunakan skor dasar langsung
                mmr_score = candidate_base_score
            else:
                # Hitung MMR dengan mempertimbangkan item yang sudah dipilih
                selected_gids = idx_all[selected_positions]
                similarity = cosine_similarity(X[item_gid], X[selected_gids]).max()
                mmr_score = lambda_mmr * candidate_base_score - (1.0 - lambda_mmr) * float(similarity)
            
            # Update kandidat terbaik
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_position = position
        
        # Jika tidak ada kandidat valid, berhenti
        if best_position is None:
            break
        
        # Tambahkan kandidat terbaik ke hasil
        selected_positions.append(best_position)
        
        # Update counter kategori
        if per_category_cap and items is not None:
            _update_category_count(int(idx_all[best_position]), items, category_count)
        
        # Hapus dari kandidat yang tersisa
        remaining_candidates.remove(best_position)
        
        # Cukup jika sudah mencapai top_n
        if len(selected_positions) >= top_n:
            break
    
    # Konversi posisi ke global ID
    return [int(idx_all[pos]) for pos in selected_positions]


def _filter_blocked_items(idx_all: np.ndarray, blocked_gids: Optional[Set[int]]) -> np.ndarray:
    """
    Filter out item yang sudah di-skip/block oleh user.
    
    Args:
        idx_all: Array index semua kandidat
        blocked_gids: Set berisi index item yang di-block
    
    Returns:
        np.ndarray: Array index tanpa item yang di-block
    """
    if not blocked_gids:
        return idx_all
    
    blocked_set = set(blocked_gids)
    return np.array([i for i in idx_all if i not in blocked_set])


def _calculate_base_scores(items: pd.DataFrame, idx_all: np.ndarray) -> np.ndarray:
    """
    Menghitung skor dasar berdasarkan rating item.
    
    Cara Kerja:
        1. Ambil rating dari item yang lolos filter
        2. Normalisasi ke range 0-1 menggunakan min-max
        3. Tambah noise kecil untuk menghindari tie (skor sama)
    
    Args:
        items: DataFrame item
        idx_all: Index item yang lolos filter
    
    Returns:
        np.ndarray: Skor dasar yang sudah dinormalisasi
    """
    # Ambil rating, fill NaN dengan 0, clip nilai negatif
    ratings = items.iloc[idx_all]["rating"].fillna(0.0).clip(lower=0.0).values.astype(float)
    
    # Normalisasi ke 0-1
    normalized_scores = normalize_minmax(ratings)
    
    # Tambah noise kecil (1e-4) untuk tie-breaking
    # RandomState(13) untuk reproducibility
    noise = np.random.RandomState(13).rand(len(idx_all)) * 1e-4
    
    return normalized_scores + noise


def _add_serendipity_items(
    selected_gids: List[int], 
    idx_all: np.ndarray, 
    items: pd.DataFrame,
    serendipity_pct: int,
    top_n: int
) -> List[int]:
    """
    Menambahkan item "serendipity" (kejutan) ke hasil rekomendasi.
    
    Tujuan:
        Menambah keragaman dengan menyisipkan beberapa item populer
        yang mungkin tidak terpilih oleh algoritma MMR, tapi bisa
        menarik bagi user sebagai "kejutan".
    
    Analogi:
        Seperti ketika Spotify menyisipkan lagu popular di playlist
        yang mungkin bukan genre favoritmu, tapi mungkin kamu suka.
    
    Args:
        selected_gids: Daftar item yang sudah dipilih oleh MMR
        idx_all: Semua kandidat yang lolos filter
        items: DataFrame items
        serendipity_pct: Persentase item serendipity (0-30%)
        top_n: Target jumlah total item
    
    Returns:
        List[int]: Daftar item yang sudah ditambahi serendipity
    """
    # Hitung jumlah item serendipity yang akan ditambahkan
    min_serendipity = 1
    max_serendipity = top_n // 5  # Maksimal 20% dari top_n
    serendipity_count = max(0, min(max_serendipity, int(len(idx_all) * serendipity_pct / 100)))
    
    if serendipity_count <= 0:
        return selected_gids
    
    # Cari item yang belum terpilih
    already_selected = set(selected_gids)
    available_pool = [g for g in idx_all if g not in already_selected]
    
    if not available_pool:
        return selected_gids
    
    # Urutkan berdasarkan rating (popularitas)
    pool_items = items.iloc[available_pool].copy()
    sorted_by_rating = list(pool_items.sort_values(["rating"], ascending=False).index)
    
    # Shuffle untuk menambah kejutan
    rng = np.random.RandomState(31)
    rng.shuffle(sorted_by_rating)
    
    # Ambil beberapa item teratas
    serendipity_picks = sorted_by_rating[:serendipity_count]
    
    # Gabungkan dengan hasil yang sudah ada
    result = selected_gids.copy()
    result.extend(serendipity_picks)
    
    return result


def build_feed_cbf(
    items: pd.DataFrame, 
    X, 
    filters: Dict, 
    top_n: int = 12, 
    mmr_lambda: float = 0.7, 
    per_category_cap: int = 2, 
    serendipity_pct: int = 15, 
    blocked_gids: Optional[Set[int]] = None
) -> List[Candidate]:
    """
    Membangun feed rekomendasi menggunakan Content-Based Filtering.
    
    Ini adalah fungsi UTAMA untuk tab Feed yang menampilkan rekomendasi
    tanpa query pencarian (berdasarkan filter dan rating saja).
    
    Alur Kerja:
        1. Filter item berdasarkan kategori/kota/harga
        2. Hapus item yang di-skip user
        3. Hitung skor dasar dari rating
        4. Pilih item menggunakan MMR (relevan + beragam)
        5. Tambahkan item serendipity (kejutan)
        6. Kembalikan hasil dalam bentuk list Candidate
    
    Args:
        items: DataFrame berisi semua destinasi wisata
        X: Matriks TF-IDF (scipy sparse matrix)
        filters: Dictionary filter user (categories, cities, max_price)
        top_n: Jumlah item yang ditampilkan (default: 12)
        mmr_lambda: Parameter MMR 0-1 (default: 0.7)
        per_category_cap: Batas item per kategori (default: 2)
        serendipity_pct: Persentase item kejutan (default: 15%)
        blocked_gids: Set index item yang di-skip user
    
    Returns:
        List[Candidate]: Daftar kandidat dengan gid dan base_score
    
    Contoh Penggunaan:
        >>> candidates = build_feed_cbf(
        ...     items=items, X=X,
        ...     filters={"categories": ["Pantai"], "cities": ["Aceh"]},
        ...     top_n=10
        ... )
        >>> for c in candidates:
        ...     print(f"Item {c.gid}: skor {c.base_score:.2f}")
    """
    # Step 1: Filter item berdasarkan kriteria user
    filter_mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[filter_mask]
    
    # Step 2: Hapus item yang di-skip/block
    idx_all = _filter_blocked_items(idx_all, blocked_gids)
    
    # Jika tidak ada item yang lolos, return empty
    if idx_all.size == 0:
        return []
    
    # Step 3: Hitung skor dasar dari rating
    base_scores = _calculate_base_scores(items, idx_all)
    
    # Step 4: Pilih item menggunakan MMR
    selected_gids = mmr_select(
        idx_all=idx_all, 
        X=X, 
        base_scores=base_scores, 
        top_n=top_n,
        lambda_mmr=mmr_lambda, 
        per_category_cap=per_category_cap, 
        items=items
    )
    
    # Step 5: Tambahkan item serendipity
    selected_gids = _add_serendipity_items(
        selected_gids=selected_gids,
        idx_all=idx_all,
        items=items,
        serendipity_pct=serendipity_pct,
        top_n=top_n
    )
    
    # Step 6: Buat mapping skor untuk output
    score_map = {int(idx_all[i]): float(base_scores[i]) for i in range(len(idx_all))}
    
    # Batasi hasil ke top_n dan buat list Candidate
    result = [
        Candidate(gid=int(g), base_score=float(score_map.get(int(g), 0.0))) 
        for g in selected_gids[:top_n]
    ]
    
    return result


def _search_with_nn_index(
    query_vector, 
    nbrs, 
    idx_all: np.ndarray, 
    top_n: int
) -> Optional[List[Tuple[int, float]]]:
    """
    Pencarian cepat menggunakan NearestNeighbors index.
    
    Keuntungan:
        Lebih cepat dari cosine_similarity penuh, terutama 
        untuk dataset besar karena menggunakan algoritma 
        approximate nearest neighbors.
    
    Args:
        query_vector: Vektor TF-IDF dari query user
        nbrs: NearestNeighbors index (dari sklearn)
        idx_all: Index item yang lolos filter
        top_n: Jumlah hasil yang diinginkan
    
    Returns:
        List[(gid, similarity)] atau None jika gagal/kosong
    """
    # Cari tetangga terdekat
    n_neighbors = min(max(100, top_n * 5), nbrs.n_samples_fit_)
    distances, neighbor_indices = nbrs.kneighbors(query_vector, n_neighbors=n_neighbors)
    
    # Konversi distance ke similarity (cosine distance -> similarity)
    similarities = 1.0 - distances[0]
    candidates = neighbor_indices[0]
    
    # Filter hanya yang ada di idx_all (lolos filter user)
    valid_indices = set(idx_all.tolist())
    results = [
        (int(gid), float(sim)) 
        for gid, sim in zip(candidates, similarities) 
        if gid in valid_indices
    ]
    
    return results if results else None


def _search_fallback(
    query_vector, 
    X, 
    idx_all: np.ndarray
) -> np.ndarray:
    """
    Pencarian fallback menggunakan cosine_similarity penuh.
    
    Digunakan ketika:
        - NearestNeighbors index tidak tersedia
        - Hasil dari NN index kosong setelah filter
    
    Args:
        query_vector: Vektor TF-IDF dari query
        X: Matriks TF-IDF lengkap
        idx_all: Index item yang lolos filter
    
    Returns:
        np.ndarray: Array similarity scores
    """
    # Hitung cosine similarity antara query dan semua item yang lolos filter
    similarities = cosine_similarity(X[idx_all], query_vector)[:, 0]
    return similarities


def search_cbf(
    items: pd.DataFrame, 
    X, 
    vectorizer, 
    nbrs, 
    query: str, 
    filters: Dict, 
    top_n: int = 12, 
    mmr_lambda: float = 0.7, 
    per_category_cap: int = 3,
    similarity_threshold: float = 0.1
) -> List[Tuple[int, float]]:
    """
    Pencarian destinasi menggunakan query text (Knowledge Base search).
    
    Ini adalah fungsi UTAMA untuk tab Search yang mencari destinasi
    berdasarkan kata kunci dari user.
    
    Alur Kerja:
        1. Validasi vectorizer tersedia
        2. Filter item berdasarkan kriteria user
        3. Preprocess dan transform query ke vektor TF-IDF
        4. Cari item mirip menggunakan NN index (atau fallback)
        5. Pilih hasil menggunakan MMR untuk diversitas
        6. Return hasil sebagai list (gid, similarity)
    
    Cara Kerja TF-IDF Search:
        - Query user diubah menjadi vektor angka (TF-IDF)
        - Setiap item juga punya vektor TF-IDF dari deskripsinya
        - Item dengan vektor paling "mirip" (cosine similarity tinggi)
          akan muncul di hasil pencarian
    
    Args:
        items: DataFrame destinasi wisata
        X: Matriks TF-IDF
        vectorizer: TF-IDF vectorizer yang sudah di-fit
        nbrs: NearestNeighbors index (opsional, bisa None)
        query: Kata kunci pencarian dari user
        filters: Filter kategori/kota/harga
        top_n: Jumlah hasil (default: 12)
        mmr_lambda: Parameter MMR (default: 0.7)
        per_category_cap: Batas per kategori (default: 3)
    
    Returns:
        List[(gid, score)]: Daftar tuple (index item, skor similarity)
    
    Contoh:
        >>> results = search_cbf(
        ...     items, X, vectorizer, nbrs,
        ...     query="pantai snorkeling aceh",
        ...     filters={},
        ...     top_n=10
        ... )
        >>> for gid, score in results:
        ...     print(f"Item {gid}: similarity {score:.3f}")
    """
    # Validasi: vectorizer harus tersedia
    if vectorizer is None:
        return []  # Tidak bisa search tanpa vectorizer
    
    # Step 1: Filter item berdasarkan kriteria
    filter_mask = _mask_by_filters(items, filters)
    idx_all = np.arange(len(items))[filter_mask]
    
    if idx_all.size == 0:
        return []  # Tidak ada item yang lolos filter
    
    # Step 2: Preprocess query dan transform ke vektor TF-IDF
    try:
        processed_query = preprocess_text(query)
        # Handle case where vectorizer might not be fitted properly
        if hasattr(vectorizer, 'idf_'):
            query_vector = vectorizer.transform([processed_query])
        else:
            # Fallback for some sklearn versions or if loaded incorrectly
            # Try to force check if it looks fitted, otherwise raise/return
            print("Warning: Vectorizer attribute 'idf_' missing. Attempting transform anyway.")
            query_vector = vectorizer.transform([processed_query])
            
    except (NotFittedError, AttributeError, ValueError) as e:
        print(f"Error in vectorizer: {e}")
        return []  # Return empty if vectorizer fails

    
    # Step 3: Cari item mirip
    if nbrs is not None:
        # Gunakan NearestNeighbors index (lebih cepat)
        nn_results = _search_with_nn_index(query_vector, nbrs, idx_all, top_n)
        
        if nn_results:
            # Filter results below threshold
            nn_results = [(g, s) for g, s in nn_results if s >= similarity_threshold]

            if not nn_results:
                return []

            # Hasil valid dari NN index
            sub_gids = np.array([g for g, _ in nn_results], dtype=int)
            sub_scores = normalize_minmax([s for _, s in nn_results])
            
            # Pilih dengan MMR untuk diversitas
            selected_gids = mmr_select(
                idx_all=sub_gids, 
                X=X, 
                base_scores=sub_scores, 
                top_n=top_n,
                lambda_mmr=mmr_lambda, 
                per_category_cap=per_category_cap, 
                items=items
            )
            
            # Buat mapping skor untuk output
            score_map = {int(g): float(s) for g, s in zip(sub_gids, sub_scores)}
            return [(int(g), float(score_map.get(int(g), 0.0))) for g in selected_gids]
    
    # Fallback: Gunakan cosine_similarity penuh
    similarities = _search_fallback(query_vector, X, idx_all)
    
    # Filter by threshold (Raw Cosine Similarity)
    # Kita hanya ambil item yang kemiripannya >= threshold (misal 0.1)
    valid_mask = similarities >= similarity_threshold
    
    if not np.any(valid_mask):
        return []
        
    # Update idx_all dan similarities hanya ke item yang valid
    idx_all = idx_all[valid_mask]
    similarities = similarities[valid_mask]
    
    base_scores = normalize_minmax(similarities)
    
    # Pilih dengan MMR
    selected_gids = mmr_select(
        idx_all=idx_all, 
        X=X, 
        base_scores=base_scores, 
        top_n=top_n,
        lambda_mmr=mmr_lambda, 
        per_category_cap=per_category_cap, 
        items=items
    )
    
    # Return hasil dengan skor
    idx_list = list(idx_all)
    return [(int(g), float(base_scores[idx_list.index(int(g))])) for g in selected_gids]
