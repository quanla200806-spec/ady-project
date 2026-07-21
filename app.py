import pickle
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ==========================================
# 1. CẤU HÌNH & HẰNG SỐ
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_FILE = BASE_DIR / "heart_cleaned.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
TABLE_NAME = "heart_data"

# Cấu hình SQL Server (Mặc định cố định trong code, không hiện ở Sidebar nữa)
DB_SERVER = "localhost"
DB_NAME = "VeraHeart"
DB_USER = ""
DB_PASSWORD = ""
USE_WINDOWS_AUTH = True

REQUIRED_COLUMNS = [
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio", "BMI"
]

# Các cột phải là số nguyên, dùng khi kiểm tra dữ liệu trước khi ghi SQL
INTEGER_COLUMNS = ["id", "age", "gender", "height", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]

# Ngưỡng làm sạch dữ liệu thô (gộp từ Clean_Tool.py vào thẳng app, không cần chạy tool riêng nữa)
AP_HI_MIN, AP_HI_MAX = 70, 250
AP_LO_MIN, AP_LO_MAX = 40, 200
HEIGHT_MIN = 120

st.set_page_config(page_title="VeraHeart - ADY201m", layout="wide")
st.title("VeraHeart - Quản lý dữ liệu bệnh nhân")

# ==========================================
# 2. KẾT NỐI & KHỞI TẠO SQL SERVER
# ==========================================
def get_engine():
    driver = "ODBC Driver 17 for SQL Server"
    if USE_WINDOWS_AUTH:
        conn_str = f"DRIVER={{{driver}}};SERVER={DB_SERVER};DATABASE={DB_NAME};Trusted_Connection=yes;TrustServerCertificate=yes;"
    else:
        conn_str = f"DRIVER={{{driver}}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD};TrustServerCertificate=yes;"
    params = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def init_db(engine):
    """Tạo bảng SQL đơn giản nếu chưa tồn tại."""
    create_table_sql = f"""
    IF OBJECT_ID(N'dbo.{TABLE_NAME}', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.{TABLE_NAME} (
            id INT PRIMARY KEY,
            age INT, gender INT, height INT, weight FLOAT,
            ap_hi INT, ap_lo INT, cholesterol INT, gluc INT,
            smoke INT, alco INT, active INT, cardio INT, BMI FLOAT
        );
    END
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

# ==========================================
# 3. HÀM XỬ LÝ DỮ LIỆU & MACHINE LEARNING
# ==========================================
def load_csv(file_path_or_buffer):
    """Đọc file CSV/Excel và tự động chuẩn hóa kiểu dữ liệu số.
    Thử nhiều encoding + tự dò dấu phân tách cột, vì file xuất từ Excel ở VN
    thường không phải UTF-8/dấu phẩy chuẩn."""
    try:
        # Với st.file_uploader, file_path_or_buffer là UploadedFile -> có .name, .size
        file_name = getattr(file_path_or_buffer, "name", str(file_path_or_buffer))
        file_size = getattr(file_path_or_buffer, "size", None)

        if file_size == 0:
            st.error(f"File '{file_name}' có dung lượng 0 byte (file rỗng). Vui lòng kiểm tra lại file gốc.")
            return None

        if file_name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path_or_buffer)
        else:
            df = None
            last_err = None
            # Thử lần lượt các encoding phổ biến (kể cả file CSV xuất từ Excel VN)
            for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
                try:
                    if hasattr(file_path_or_buffer, "seek"):
                        file_path_or_buffer.seek(0)
                    # sep=None + engine="python" để pandas TỰ DÒ dấu phân tách (, ; \t ...)
                    tmp = pd.read_csv(file_path_or_buffer, sep=None, engine="python", encoding=encoding)
                    if tmp.shape[1] > 1:  # đọc ra nhiều hơn 1 cột -> coi như thành công
                        df = tmp
                        break
                    last_err = ValueError(f"Chỉ đọc được 1 cột với encoding {encoding}")
                except Exception as e:
                    last_err = e

            if df is None:
                raise ValueError(
                    f"Không đọc được dữ liệu từ file '{file_name}' (size={file_size} bytes). "
                    f"Lỗi cuối cùng: {last_err}"
                )

        for col in df.columns:
            if col in REQUIRED_COLUMNS:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def clean_raw_data(df):
    """Làm sạch dữ liệu THÔ ngay trong app (thay cho việc chạy Clean_Tool.py riêng):
    - Tự tính cột BMI = weight / (height/100)^2 nếu chưa có / bị thiếu.
    - Loại các dòng outlier: ap_hi, ap_lo ngoài ngưỡng hợp lý, ap_lo > ap_hi,
      height quá thấp (< 120cm) — giống logic gốc trong Clean_Tool.py.
    Trả về (clean_df, outliers_df) để hiển thị cho người dùng xem trước khi đồng bộ."""
    df = df.copy()

    # Tính BMI nếu chưa có cột hoặc còn thiếu giá trị
    if "height" in df.columns and "weight" in df.columns:
        need_bmi = "BMI" not in df.columns or df["BMI"].isna().any()
        if need_bmi:
            calc_bmi = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
            if "BMI" in df.columns:
                df["BMI"] = df["BMI"].fillna(calc_bmi)
            else:
                df["BMI"] = calc_bmi

    outlier_mask = pd.Series(False, index=df.index)
    if "ap_hi" in df.columns:
        outlier_mask |= (df["ap_hi"] < AP_HI_MIN) | (df["ap_hi"] > AP_HI_MAX)
    if "ap_lo" in df.columns:
        outlier_mask |= (df["ap_lo"] < AP_LO_MIN) | (df["ap_lo"] > AP_LO_MAX)
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        outlier_mask |= (df["ap_lo"] > df["ap_hi"])
    if "height" in df.columns:
        outlier_mask |= (df["height"] < HEIGHT_MIN)

    outliers_df = df[outlier_mask].copy()
    clean_df = df[~outlier_mask].copy()
    return clean_df, outliers_df


def validate_patient_data(df):
    """Kiểm tra dữ liệu tối thiểu trước khi ghi SQL: đủ cột, đúng kiểu số,
    không trùng ID, cardio chỉ được 0 hoặc 1. Dòng lỗi sẽ bị tách riêng
    (skipped_df) thay vì âm thầm bỏ qua."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError("Thiếu cột bắt buộc trong dữ liệu: " + ", ".join(missing_columns))

    clean_df = df[REQUIRED_COLUMNS].copy()
    invalid_mask = pd.Series(False, index=clean_df.index)

    # Ép kiểu số cho tất cả các cột bắt buộc, giá trị không hợp lệ -> NaN
    for col in REQUIRED_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        invalid_mask |= clean_df[col].isna()

    # ID không được trùng nhau trong cùng 1 lần đồng bộ
    invalid_mask |= clean_df["id"].notna() & clean_df["id"].duplicated(keep=False)

    # cardio chỉ được phép là 0 hoặc 1
    invalid_mask |= clean_df["cardio"].notna() & ~clean_df["cardio"].isin([0, 1])

    skipped_df = clean_df[invalid_mask].copy()
    valid_df = clean_df[~invalid_mask].copy()

    for col in INTEGER_COLUMNS:
        valid_df[col] = valid_df[col].astype(int)
    valid_df["weight"] = valid_df["weight"].astype(float)
    valid_df["BMI"] = valid_df["BMI"].astype(float)

    return valid_df, skipped_df

def upsert_patient_sql(conn, patient_dict):
    """Thêm hoặc cập nhật (UPSERT) 1 bệnh nhân, dùng chung 1 connection/transaction
    do nơi gọi cung cấp (không tự mở transaction riêng nữa)."""
    merge_sql = f"""
    MERGE dbo.{TABLE_NAME} AS target
    USING (SELECT :id AS id) AS source ON (target.id = source.id)
    WHEN MATCHED THEN
        UPDATE SET age=:age, gender=:gender, height=:height, weight=:weight,
                   ap_hi=:ap_hi, ap_lo=:ap_lo, cholesterol=:cholesterol, gluc=:gluc,
                   smoke=:smoke, alco=:alco, active=:active, cardio=:cardio, BMI=:BMI
    WHEN NOT MATCHED THEN
        INSERT (id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio, BMI)
        VALUES (:id, :age, :gender, :height, :weight, :ap_hi, :ap_lo, :cholesterol, :gluc, :smoke, :alco, :active, :cardio, :BMI);
    """
    conn.execute(text(merge_sql), patient_dict)

@st.cache_resource
def load_ml_model():
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# ==========================================
# 4. GIAO DIỆN CHÍNH (5 TABS)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 CSV & Đồng bộ", "➕ Nhập bệnh nhân", "🔍 Tìm kiếm", "🤖 Dự đoán ML"
])

# ----- TAB 1: CSV & ĐỒNG BỘ -----
with tab1:
    st.subheader("Xem Dữ Liệu & Đồng Bộ SQL")
    uploaded_file = st.file_uploader("Upload file CSV/Excel mới (để trống nếu dùng file mặc định)", type=["csv", "xlsx"])
    
    source = uploaded_file if uploaded_file else (LOCAL_DATA_FILE if LOCAL_DATA_FILE.exists() else None)
    
    if source:
        df_raw = load_csv(source)
        if df_raw is not None:
            st.markdown("**Dữ liệu thô (trước khi làm sạch)**")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.caption(f"Tổng số dòng (thô): {len(df_raw):,}")

            # Làm sạch dữ liệu ngay trong app: tính BMI + loại outlier (thay cho Clean_Tool.py)
            df, outliers_df = clean_raw_data(df_raw)
            st.markdown("**Dữ liệu sau khi làm sạch**")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(
                f"Sau khi làm sạch: {len(df):,} dòng hợp lệ, "
                f"{len(outliers_df):,} dòng bị loại (ap_hi/ap_lo/height bất thường)."
            )
            if len(outliers_df):
                with st.expander(f"Xem {len(outliers_df):,} dòng bị loại do outlier"):
                    st.dataframe(outliers_df, use_container_width=True)

            # Đồng bộ dữ liệu (đã làm sạch) vào SQL
            if st.button("Đồng bộ dữ liệu CSV này vào SQL Server", use_container_width=True):
                try:
                    valid_df, skipped_df = validate_patient_data(df)
                    engine = get_engine()
                    init_db(engine)

                    with engine.begin() as conn:  # 1 transaction duy nhất cho cả batch
                        for _, row in valid_df.iterrows():
                            upsert_patient_sql(conn, row.to_dict())

                    st.success(f"Đã đồng bộ thành công {len(valid_df):,} dòng vào SQL Server!")
                    if len(skipped_df):
                        st.warning(f"Đã bỏ qua {len(skipped_df):,} dòng dữ liệu không hợp lệ (thiếu/sai kiểu/trùng ID/cardio sai).")
                        with st.expander("Xem các dòng bị bỏ qua"):
                            st.dataframe(skipped_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi đồng bộ (đã rollback, dữ liệu chưa lưu vào SQL): {e}")

# ----- TAB 2: NHẬP BỆNH NHÂN -----
with tab2:
    st.subheader("Nhập thông tin bệnh nhân mới")
    with st.form("add_patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_id = st.number_input("ID", min_value=1, value=1)
            p_age = st.number_input("Tuổi", min_value=1, max_value=120, value=40)
            p_gender = st.selectbox("Giới tính", ["Nữ", "Nam"])
            p_height = st.number_input("Chiều cao (cm)", min_value=50, max_value=250, value=165)
            p_weight = st.number_input("Cân nặng (kg)", min_value=10.0, max_value=200.0, value=60.0)
        with c2:
            p_ap_hi = st.number_input("Huyết áp tâm thu (ap_hi)", min_value=50, max_value=250, value=120)
            p_ap_lo = st.number_input("Huyết áp tâm trương (ap_lo)", min_value=30, max_value=180, value=80)
            p_chol = st.selectbox("Cholesterol", ["Bình thường", "Cao", "Rất cao"])
            p_gluc = st.selectbox("Glucose", ["Bình thường", "Cao", "Rất cao"])
        with c3:
            p_smoke = st.selectbox("Hút thuốc", ["Không", "Có"])
            p_alco = st.selectbox("Uống rượu", ["Không", "Có"])
            p_active = st.selectbox("Vận động", ["Không", "Có"])
            p_cardio = st.selectbox("Bệnh tim", ["Không", "Có"])
        
        bmi = round(p_weight / ((p_height / 100) ** 2), 2)
        st.metric("BMI tự động tính", bmi)
        
        submitted = st.form_submit_button("Lưu bệnh nhân vào SQL")

    if submitted:
        patient_data = {
            "id": int(p_id), "age": int(p_age),
            "gender": 1 if p_gender == "Nữ" else 2,
            "height": int(p_height), "weight": float(p_weight),
            "ap_hi": int(p_ap_hi), "ap_lo": int(p_ap_lo),
            "cholesterol": ["Bình thường", "Cao", "Rất cao"].index(p_chol) + 1,
            "gluc": ["Bình thường", "Cao", "Rất cao"].index(p_gluc) + 1,
            "smoke": 1 if p_smoke == "Có" else 0,
            "alco": 1 if p_alco == "Có" else 0,
            "active": 1 if p_active == "Có" else 0,
            "cardio": 1 if p_cardio == "Có" else 0,
            "BMI": float(bmi)
        }
        try:
            valid_df, skipped_df = validate_patient_data(pd.DataFrame([patient_data]))
            if len(skipped_df):
                st.error("Dữ liệu bệnh nhân không hợp lệ (kiểm tra lại ID/cardio), chưa ghi vào SQL.")
            else:
                engine = get_engine()
                init_db(engine)
                with engine.begin() as conn:
                    upsert_patient_sql(conn, valid_df.iloc[0].to_dict())
                st.success(f"Lưu bệnh nhân ID {p_id} vào SQL Server thành công!")
        except Exception as e:
            st.error(f"Lỗi lưu SQL (đã rollback nếu có lỗi): {e}")

# ----- TAB 3: TÌM KIẾM BỆNH NHÂN -----
with tab3:
    st.subheader("Tìm kiếm bệnh nhân theo ID")
    search_id = st.number_input("Nhập ID cần tìm", min_value=1, value=1)
    if st.button("Tìm kiếm"):
        try:
            engine = get_engine()
            init_db(engine)
            with engine.connect() as conn:
                res = pd.read_sql(text(f"SELECT * FROM dbo.{TABLE_NAME} WHERE id = :id"), conn, params={"id": search_id})
            if not res.empty:
                st.success(f"Tìm thấy bệnh nhân ID {search_id}")
                st.dataframe(res, use_container_width=True)
            else:
                st.warning("Không tìm thấy bệnh nhân với ID này.")
        except Exception as e:
            st.error(f"Lỗi tìm kiếm: {e}")

# ----- TAB 4: DỰ ĐOÁN MACHINE LEARNING -----
with tab4:
    st.subheader("🤖 Dự đoán nguy cơ mắc bệnh tim (ML)")
    ml_data = load_ml_model()
    
    if ml_data is None:
        st.error("Chưa tìm thấy file model.pkl. Vui lòng kiểm tra lại file mô hình!")
    else:
        st.info(f"Mô hình: **{ml_data.get('model_name', 'RandomForest')}** | Độ chính xác: **{ml_data.get('accuracy', 0)*100:.2f}%**")
        
        with st.form("ml_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ml_age = st.number_input("Tuổi", min_value=1, max_value=120, value=50, key="ml_age")
                ml_gender = st.selectbox("Giới tính", ["Nữ", "Nam"], key="ml_gender")
                ml_height = st.number_input("Chiều cao (cm)", min_value=50, max_value=250, value=165, key="ml_h")
                ml_weight = st.number_input("Cân nặng (kg)", min_value=10.0, max_value=200.0, value=70.0, key="ml_w")
            with col2:
                ml_ap_hi = st.number_input("Huyết áp tâm thu (ap_hi)", min_value=50, max_value=250, value=120, key="ml_hi")
                ml_ap_lo = st.number_input("Huyết áp tâm trương (ap_lo)", min_value=30, max_value=180, value=80, key="ml_lo")
                ml_chol = st.selectbox("Cholesterol", ["Bình thường", "Cao", "Rất cao"], key="ml_chol")
                ml_gluc = st.selectbox("Glucose", ["Bình thường", "Cao", "Rất cao"], key="ml_gluc")
            with col3:
                ml_smoke = st.selectbox("Hút thuốc", ["Không", "Có"], key="ml_smoke")
                ml_alco = st.selectbox("Uống rượu", ["Không", "Có"], key="ml_alco")
                ml_active = st.selectbox("Vận động", ["Không", "Có"], key="ml_active")
                
            ml_predict_btn = st.form_submit_button("🔍 Dự đoán ngay", use_container_width=True)

        if ml_predict_btn:
            ml_bmi = round(ml_weight / ((ml_height / 100) ** 2), 2)

            feature_values = [
                int(ml_age),
                1 if ml_gender == "Nữ" else 2,
                int(ml_height),
                float(ml_weight),
                int(ml_ap_hi),
                int(ml_ap_lo),
                ["Bình thường", "Cao", "Rất cao"].index(ml_chol) + 1,
                ["Bình thường", "Cao", "Rất cao"].index(ml_gluc) + 1,
                1 if ml_smoke == "Có" else 0,
                1 if ml_alco == "Có" else 0,
                1 if ml_active == "Có" else 0,
                float(ml_bmi),
            ]
            features_df = pd.DataFrame([feature_values], columns=ml_data.get("features", [
                "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active", "BMI"
            ]))

            scaler = ml_data.get("scaler")
            imputer = ml_data.get("imputer")
            model = ml_data["model"]

            features_processed = features_df
            if imputer is not None:
                features_processed = imputer.transform(features_df)

            if scaler is not None:
                features_scaled = scaler.transform(features_processed)
            else:
                features_scaled = features_processed

            probs = model.predict_proba(features_scaled)[0]
            prob_healthy = probs[0] * 100
            prob_sick = probs[1] * 100
            
            st.write("---")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Không mắc bệnh tim", f"{prob_healthy:.2f}%")
            res_col2.metric("Có nguy cơ mắc bệnh tim", f"{prob_sick:.2f}%")
            
            if prob_sick >= 50:
                st.warning("⚠️ **Kết luận:** Người này có nguy cơ cao mắc bệnh tim!")
            else:
                st.success("✅ **Kết luận:** Người này có nguy cơ thấp / bình thường.")

#Streamlit run app.py
