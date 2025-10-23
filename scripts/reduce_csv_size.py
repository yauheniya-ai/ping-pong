import pandas as pd

df = pd.read_csv("scripts/demo_results/transitions_20251023_101839.csv")
df = pd.concat([df.iloc[[0]], df.head(100)], ignore_index=True)
df.to_csv("scripts/transitions_trimmed_head.csv", index=False)