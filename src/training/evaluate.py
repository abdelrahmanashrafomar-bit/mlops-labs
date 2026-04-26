from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import json

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]

    data_path = BASE_DIR / "data" / "processed" / "train.csv"
    model_path = BASE_DIR / "models" / "model.pkl"

    # 📥 load
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    print(f"Final accuracy: {acc}")
    Path("reports").mkdir(exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)



if __name__ == "__main__":
    main()
