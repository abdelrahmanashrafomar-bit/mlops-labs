from pathlib import Path

import pandas as pd


def main():
    # 📁 root
    BASE_DIR = Path(__file__).resolve().parents[2]

    # 📂 paths
    raw_path = BASE_DIR / "data" / "raw" / "train.csv"
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    processed_path = processed_dir / "train.csv"

    # 📥 load data
    df = pd.read_csv(raw_path)

    # 🧼 preprocessing
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # ✂️ features + target
    X = df.drop(columns=["Survived", "Name", "Ticket", "Cabin"])
    y = df["Survived"]

    processed = X.copy()
    processed["Survived"] = y

    # 💾 save
    processed.to_csv(processed_path, index=False)

    print(f"Processed data saved at: {processed_path}")


if __name__ == "__main__":
    main()
