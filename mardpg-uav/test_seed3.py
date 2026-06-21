import numpy as np

rng = np.random.RandomState(42)
print(f"Seed 42: {sorted(rng.choice(25, 16, replace=False))}")
rng.seed(43)
print(f"Seed 43: {sorted(rng.choice(25, 16, replace=False))}")
rng.seed(44)
print(f"Seed 44: {sorted(rng.choice(25, 16, replace=False))}")
