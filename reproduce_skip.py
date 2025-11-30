
import pandas as pd
import numpy as np
from eco_recsys.cbf import build_feed_cbf

# Mock data
items = pd.DataFrame({
    "category": ["Nature", "City", "Nature", "Beach"],
    "city": ["CityA", "CityB", "CityA", "CityC"],
    "price": [10000, 20000, 15000, 30000],
    "rating": [4.5, 4.0, 4.8, 3.5]
})
X = np.random.rand(4, 10) # Mock embeddings

filters = {}

# Test 1: No blocked items
print("Test 1: No blocked items")
feed = build_feed_cbf(items, X, filters, top_n=4, blocked_gids=None)
print(f"Feed IDs: {[c.gid for c in feed]}")
assert len(feed) == 4, "Should return all 4 items"

# Test 2: Block item 0
print("\nTest 2: Block item 0")
blocked = {0}
feed_blocked = build_feed_cbf(items, X, filters, top_n=4, blocked_gids=blocked)
print(f"Feed IDs (blocked={blocked}): {[c.gid for c in feed_blocked]}")
assert 0 not in [c.gid for c in feed_blocked], "Item 0 should be blocked"
assert len(feed_blocked) == 3, "Should return 3 items"

print("\nSUCCESS: Skip logic verified.")
