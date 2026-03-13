import pandas as pd

train = pd.read_csv("data/processed/train_features.csv")
new = pd.read_csv("data/processed/new_features.csv")

for col in train.columns:
    if train[col].dtype != "object":
        diff = abs(train[col].mean() - new[col].mean())
        print(col, diff)