import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python view_npz_shape.py <file.npz>")
    sys.exit(1)

npz_path = sys.argv[1]
data = np.load(npz_path)

print(f"Keys in {npz_path}:")
for key in data.files:
    arr = data[key]
    print(f"  '{key}': shape={arr.shape}, dtype={arr.dtype}")
