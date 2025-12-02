import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path so we can import eco_recsys
sys.path.append(str(Path.cwd()))

from eco_recsys.data import load_artifacts
from eco_recsys.cbf import search_cbf
from eco_recsys.utils import normalize_minmax

def verify_search():
    print("Loading artifacts...")
    art_dir = Path("artifacts")
    if not art_dir.exists():
        print(f"Artifacts directory not found at {art_dir.absolute()}")
        return

    items, X, vectorizer, nbrs = load_artifacts(art_dir)
    print(f"Loaded {len(items)} items.")


    # Test Queries
    queries = ["taman", "sungai"]

    # Debug Vectorizer
    if vectorizer:
        print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"'taman' in vocab: {'taman' in vectorizer.vocabulary_}")
        print(f"'sungai' in vocab: {'sungai' in vectorizer.vocabulary_}")
    
    from eco_recsys.text import preprocess_text

    for q in queries:
        print(f"\n--- Searching for '{q}' ---")
        processed_q = preprocess_text(q)
        print(f"Processed query: '{processed_q}'")
        
        if vectorizer:
            qv = vectorizer.transform([processed_q])
            print(f"Query vector non-zero count: {qv.nnz}")
            if qv.nnz == 0:
                print("WARNING: Query vector is empty!")

        results = search_cbf(
            items=items, X=X, vectorizer=vectorizer, nbrs=nbrs,
            query=q, filters={}, top_n=5
        )
        
        if not results:
            print("No results found.")
            continue
            
        print(f"Found {len(results)} results:")
        for gid, score in results:
            row = items.iloc[gid]
            name = row['place_name']
            desc = str(row.get('place_description', ''))
            gabungan = str(row.get('gabungan', ''))
            
            print(f"[{gid}] {name} (Score: {score:.4f})")
            
            # Check if query term is in content
            in_name = q.lower() in str(name).lower()
            in_desc = q.lower() in desc.lower()
            in_gab = q.lower() in gabungan.lower()
            print(f"    Match: Name={in_name}, Desc={in_desc}, Gabungan={in_gab}")

    # Check normalize_minmax specifically with single value
    print("\n--- Testing normalize_minmax ---")
    print(f"normalize_minmax([0.5]) = {normalize_minmax([0.5])}")
    print(f"normalize_minmax([0.5, 0.5]) = {normalize_minmax([0.5, 0.5])}")

if __name__ == "__main__":
    verify_search()
