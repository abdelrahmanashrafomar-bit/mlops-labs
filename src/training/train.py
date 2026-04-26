from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    BASE_DIR = Path(__file__).resolve().parents[2]

    data_path = BASE_DIR / "data" / "processed" / "train.csv"
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    # 🔥 اختيار الموديل من config
    if cfg.model == "log_reg":
        model = LogisticRegression(max_iter=500)
    elif cfg.model == "rf":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Unknown model")

    model.fit(X_train, y_train)

    score = model.score(X_val, y_val)
    print(f"Model: {cfg.model} | Accuracy: {score}")

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
