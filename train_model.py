"""
VeraHeart - Train Machine Learning Model
=========================================
Huấn luyện các mô hình ML để dự đoán nguy cơ mắc bệnh tim.
Mô hình tốt nhất sẽ được lưu thành model.pkl.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# CẤU HÌNH
# ==============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"

DATA_FILE = BASE_DIR / "cardio_train.csv"
if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR.parent / "cardio_train.csv"
if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR / "heart_cleaned.csv"

FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "BMI",
]
TARGET = "cardio"


def load_and_preprocess(csv_path):
    """Đọc dữ liệu CSV thô và tiền xử lý cho việc huấn luyện."""
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Tạo cột BMI nếu chưa có
    # Chuyển đổi kiểu dữ liệu trước khi tính BMI
    for col in ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "BMI" not in df.columns:
        df["BMI"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)

    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Giữ lại các dòng có target hợp lệ
    df = df[df[TARGET].isin([0, 1])]

    # Loại bỏ các dòng quá bất thường nhưng không loại quá nhiều
    df = df[(df["ap_hi"] > 0) & (df["ap_hi"] < 300)]
    df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 220)]
    df = df[(df["height"] > 100) & (df["height"] < 250)]
    df = df[(df["weight"] > 20) & (df["weight"] < 250)]

    # Giữ lại các hàng có đủ features và target
    df = df.dropna(subset=FEATURES + [TARGET])

    print(f"Số lượng mẫu sau tiền xử lý: {len(df)}")
    print(f"Phân phối target:\n{df[TARGET].value_counts()}")

    return df


def train_and_evaluate(df):
    """Huấn luyện nhiều mô hình mạnh hơn, so sánh Accuracy và chọn mô hình tốt nhất."""

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nSố mẫu Train: {len(X_train)}")
    print(f"Số mẫu Test:  {len(X_test)}")

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.8,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}

    print("\n" + "=" * 60)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "cv_mean": cv_scores.mean(),
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV mean accuracy: {cv_scores.mean():.4f}")
        print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Có bệnh"]))

    best_name = max(
        results,
        key=lambda k: (results[k]["accuracy"] + results[k]["cv_mean"] + results[k]["roc_auc"]) / 3,
    )
    best_model = results[best_name]["model"]
    best_accuracy = results[best_name]["accuracy"]

    print("=" * 60)
    print(f"🏆 Mô hình tốt nhất: {best_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print("=" * 60)

    return best_model, scaler, imputer, best_name, best_accuracy, results


def save_model(model, scaler, imputer, model_name, accuracy, model_path):
    """Lưu model, scaler, imputer và metadata vào file .pkl."""
    data = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "model_name": model_name,
        "accuracy": accuracy,
        "features": FEATURES,
    }
    with open(model_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✅ Đã lưu model thành công: {model_path}")
    print(f"   Model: {model_name}")
    print(f"   Accuracy: {accuracy:.4f}")


def main():
    print("=" * 60)
    print("VeraHeart - Huấn luyện mô hình Machine Learning")
    print("=" * 60)

    # 1. Đọc và tiền xử lý dữ liệu
    print("\n📊 Đang đọc và tiền xử lý dữ liệu...")
    df = load_and_preprocess(DATA_FILE)

    # 2. Huấn luyện và đánh giá
    print("\n🤖 Đang huấn luyện các mô hình...")
    best_model, scaler, imputer, best_name, best_accuracy, results = train_and_evaluate(df)

    # 3. Lưu model
    print("\n💾 Đang lưu model...")
    save_model(best_model, scaler, imputer, best_name, best_accuracy, MODEL_FILE)

    # 4. Bảng so sánh
    print("\n📋 BẢNG SO SÁNH ACCURACY:")
    print("-" * 40)
    print(f"{'Mô hình':<25} {'Accuracy':>10}")
    print("-" * 40)
    for name, result in results.items():
        marker = " ← Best" if name == best_name else ""
        print(f"{name:<25} {result['accuracy']:>9.4f}{marker}")
    print("-" * 40)


if __name__ == "__main__":
    main()
