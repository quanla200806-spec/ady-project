"""
VeraHeart - Train Machine Learning Model
=========================================
Huấn luyện các mô hình ML để dự đoán nguy cơ mắc bệnh tim.
Mô hình tốt nhất sẽ được lưu thành model.pkl.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# CẤU HÌNH
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "heart_cleaned.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

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
    """Đọc dữ liệu bằng Pandas và tiền xử lý."""
    # Đọc CSV với dấu phân cách ';' và dấu thập phân ','
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # Loại bỏ cột id (không dùng để huấn luyện)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Chuyển tất cả features và target sang kiểu số
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại bỏ dòng có giá trị thiếu trong features hoặc target
    df = df.dropna(subset=FEATURES + [TARGET])

    # Đảm bảo target chỉ có giá trị 0 hoặc 1
    df = df[df[TARGET].isin([0, 1])]

    # Loại bỏ outlier rõ ràng (ví dụ huyết áp, BMI bất thường)
    df = df[(df["ap_hi"] > 0) & (df["ap_hi"] < 300)]
    df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 200)]
    df = df[(df["height"] > 100) & (df["height"] < 250)]
    df = df[(df["weight"] > 20) & (df["weight"] < 300)]

    print(f"Số lượng mẫu sau tiền xử lý: {len(df)}")
    print(f"Phân phối target:\n{df[TARGET].value_counts()}")

    return df


def train_and_evaluate(df):
    """Huấn luyện 3 mô hình, so sánh Accuracy và chọn mô hình tốt nhất."""

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Chia dữ liệu: Train 80%, Test 20%, random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nSố mẫu Train: {len(X_train)}")
    print(f"Số mẫu Test:  {len(X_test)}")

    # Chuẩn hóa dữ liệu (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Định nghĩa các mô hình
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    print("\n" + "=" * 60)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)

    for name, model in models.items():
        # Huấn luyện
        model.fit(X_train_scaled, y_train)

        # Dự đoán
        y_pred = model.predict(X_test_scaled)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc}

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Có bệnh"]))

    # Chọn mô hình tốt nhất
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_accuracy = results[best_name]["accuracy"]

    print("=" * 60)
    print(f"🏆 Mô hình tốt nhất: {best_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print("=" * 60)

    return best_model, scaler, best_name, best_accuracy, results


def save_model(model, scaler, model_name, accuracy, model_path):
    """Lưu model, scaler và metadata vào file .pkl."""
    data = {
        "model": model,
        "scaler": scaler,
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
    best_model, scaler, best_name, best_accuracy, results = train_and_evaluate(df)

    # 3. Lưu model
    print("\n💾 Đang lưu model...")
    save_model(best_model, scaler, best_name, best_accuracy, MODEL_FILE)

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
