# parameters.py

import os

# Clustering strategy: choose between "energy" and "energyandtraintime"
CLUSTERING = os.getenv("CLUSTERING", "energy")  # Default: "energy"

# k1 allocation strategy: choose between "True", "False", "Uniform"
K1_ALLOCATION = os.getenv("K1_ALLOCATION", "reversed")  # Default: "reversed"

print(f"Using clustering method: {CLUSTERING}")
print(f"K1 allocation strategy: {K1_ALLOCATION}")
