
import numpy as np
from eco_recsys.utils import normalize_minmax

def test_normalize():
    print("Testing normalize_minmax...")
    
    # Case 1: Multiple different values
    a = [0.1, 0.5, 0.9]
    norm_a = normalize_minmax(a)
    print(f"Input: {a}")
    print(f"Output: {norm_a}")
    
    # Case 2: Single value
    b = [0.8]
    norm_b = normalize_minmax(b)
    print(f"Input: {b}")
    print(f"Output: {norm_b}")
    
    # Case 3: Multiple identical values
    c = [0.5, 0.5, 0.5]
    norm_c = normalize_minmax(c)
    print(f"Input: {c}")
    print(f"Output: {norm_c}")

    # Case 4: All zeros
    d = [0.0, 0.0]
    norm_d = normalize_minmax(d)
    print(f"Input: {d}")
    print(f"Output: {norm_d}")

if __name__ == "__main__":
    test_normalize()
