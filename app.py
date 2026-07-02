import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# ============================================================
# 0) CẤU HÌNH TRANG (luôn là lệnh Streamlit đầu tiên)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Chấm điểm tín dụng 5C - Dự báo rủi ro vỡ nợ",
    page_icon="📊",
)

# ============================================================
# HẰNG SỐ NGHIỆP VỤ (suy trực tiếp từ notebook)
# ============================================================
FEATURE_GROUPS = {
    "Tư cách (TC)": ["TC1", "TC2", "TC3", "TC4", "TC5"],
    "Năng lực (NL)": ["NL1", "NL2", "NL3", "NL4"],
    "Điều kiện (DK)": ["DK1", "DK2", "DK3", "DK4", "DK5"],
    "Vốn (V)": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "Tài sản đảm bảo (TS)": ["TS1", "TS2", "TS3", "TS4"],
}
FEATURE_COLS = [c for group in FEATURE_GROUPS.values() for c in group]
TARGET_COL = "PD"
TARGET_LABELS = {0: "Không vỡ nợ (0)", 1: "Vỡ nợ (1)"}

# ============================================================
# HÀM NẠP DỮ LIỆU DÙNG CHUNG (cache theo bytes để hashable)
# ============================================================
@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame):
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    return missing


# ============================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải tệp dữ liệu khảo sát (.csv)",
        type=["csv"],
        help="Tệp CSV chứa 24 biến khảo sát theo khung 5C và cột nhãn 'PD' (0 = không vỡ nợ, 1 = vỡ nợ).",
    )

    st.subheader("Tham số mô hình AI")
    st.caption("Mô hình: Hồi quy Logistic (Logistic Regression) - phân loại nhị phân PD.")

    C_param = st.slider(
        "C (nghịch đảo độ mạnh điều chuẩn)",
        min_value=0.01, max_value=10.0, value=1.0, step=0.01,
        help="Giá trị càng nhỏ, mô hình càng bị điều chuẩn (regularize) mạnh hơn. Mặc định scikit-learn = 1.0.",
    )
    solver = st.selectbox(
        "Solver",
        options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
        index=0,
        help="Thuật toán tối ưu hoá để huấn luyện Logistic Regression. Mặc định notebook dùng 'lbfgs'.",
    )
    max_iter = st.number_input(
        "Số vòng lặp tối đa (max_iter)",
        min_value=50, max_value=2000, value=100, step=50,
        help="Số vòng lặp tối đa để thuật toán hội tụ. Mặc định scikit-learn = 100.",
    )

    with st.expander("Tham số nâng cao"):
        test_size = st.slider(
            "Tỷ lệ tập kiểm tra (test_size)",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            help="Tỷ lệ dữ liệu dùng để đánh giá mô hình. Notebook gốc dùng 0.2.",
        )
        split_random_state = st.number_input(
            "random_state (chia tập train/test)",
            min_value=0, max_value=9999, value=23, step=1,
            help="Hạt giống ngẫu nhiên khi chia tập train/test. Notebook gốc dùng 23.",
        )
        model_random_state = st.number_input(
            "random_state (mô hình)",
            min_value=0, max_value=9999, value=23, step=1,
            help="Hạt giống ngẫu nhiên cho solver ngẫu nhiên (sag/saga). Không ảnh hưởng tới lbfgs/liblinear.",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện & Kiểm định mô hình",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ============================================================
st.title("📊 Chấm điểm tín dụng theo khung 5C - Dự báo xác suất vỡ nợ (PD)")
st.caption(
    "Ứng dụng huấn luyện mô hình Hồi quy Logistic để dự báo khả năng vỡ nợ (PD) của khách hàng "
    "dựa trên 24 câu hỏi khảo sát thuộc 5 nhóm tiêu chí tín dụng: Tư cách (TC), Năng lực (NL), "
    "Điều kiện (DK), Vốn (V) và Tài sản đảm bảo (TS), thang điểm Likert 1-5. "
    "Vui lòng tải lên tệp CSV có cấu trúc tương tự dữ liệu mẫu ở thanh bên."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu CSV ở thanh bên để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"❌ Không thể đọc tệp dữ liệu. Vui lòng kiểm tra định dạng CSV. Chi tiết lỗi: {e}")
    st.stop()

if df.empty:
    st.error("❌ Tệp dữ liệu rỗng, vui lòng tải lên tệp khác.")
    st.stop()

missing_cols = validate_columns(df)
if missing_cols:
    st.error(
        "❌ Tệp dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_cols) +
        ". Vui lòng kiểm tra lại cấu trúc tệp (24 biến khảo sát + cột 'PD')."
    )
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"Số dòng: **{df.shape[0]}** | Số cột: **{df.shape[1]}** | "
    f"Tỷ lệ vỡ nợ (PD=1) trong dữ liệu: **{df[TARGET_COL].mean() * 100:.1f}%**"
)
st.divider()

# ============================================================
# KHỐI HUẤN LUYỆN (chỉ chạy khi bấm nút, lưu vào session_state)
# ============================================================
if train_clicked:
    try:
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        if y.nunique() < 2:
            st.error("❌ Biến mục tiêu 'PD' chỉ có một lớp duy nhất, không thể huấn luyện phân loại.")
            st.stop()

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(split_random_state)
        )

        model = LogisticRegression(
            C=C_param,
            solver=solver,
            max_iter=int(max_iter),
            random_state=int(model_random_state),
        )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]

        results_df = x_test.copy()
        results_df["Thực tế (PD)"] = y_test.values
        results_df["Dự báo (PD)"] = y_pred
        results_df["Xác suất vỡ nợ"] = y_proba

        st.session_state["model"] = model
        st.session_state["preprocessor"] = None  # notebook không dùng scaler/encoder
        st.session_state["results_df"] = results_df
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["y_proba"] = y_proba
        st.session_state["feature_cols"] = FEATURE_COLS
        st.session_state["train_data_ref"] = df  # để lấy min/max/trung vị cho tab "Sử dụng mô hình"

        st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định mô hình'.")
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

# ============================================================
# CÁC TAB CHÍNH
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Tổng quan dữ liệu", "📈 Trực quan hóa dữ liệu", "🎯 Kết quả huấn luyện & kiểm định", "🔮 Sử dụng mô hình"]
)

# ------------------------------------------------------------
# THÀNH PHẦN 3: TAB TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    size_mb = len(file_bytes) / (1024 * 1024)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])
    c3.metric("Dung lượng tệp", f"{size_mb:.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình")
    st.caption("Bao gồm 24 biến đầu vào (thang Likert 1-5) và biến mục tiêu PD.")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)

# ------------------------------------------------------------
# THÀNH PHẦN 4: TAB TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------
with tab2:
    st.subheader("Trực quan hóa biến mục tiêu và biến đầu vào")

    # Chọn 3 biến đầu vào ưu tiên mặc định (tương quan tuyệt đối cao nhất với PD)
    try:
        corr_with_target = df[FEATURE_COLS].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
        default_features = corr_with_target.head(3).index.tolist()
    except Exception:
        default_features = FEATURE_COLS[:3]

    selected_features = st.multiselect(
        "Chọn thêm biến đầu vào để trực quan hóa (biến mục tiêu PD luôn hiển thị đầu tiên)",
        options=FEATURE_COLS,
        default=default_features,
        max_selections=3,
        help="Do có 24 biến đầu vào, vui lòng chọn tối đa 3 biến để hiển thị cùng biến mục tiêu PD.",
    )

    plot_vars = [TARGET_COL] + selected_features
    plot_vars = plot_vars[:4]  # tối đa 4 biểu đồ, lưới 2x2

    rows = [plot_vars[i:i + 2] for i in range(0, len(plot_vars), 2)]
    for row_vars in rows:
        cols = st.columns(2)
        for col, var in zip(cols, row_vars):
            with col:
                if var == TARGET_COL:
                    vc = df[TARGET_COL].map(TARGET_LABELS).value_counts().reset_index()
                    vc.columns = ["Nhãn", "Số lượng"]
                    fig = px.bar(
                        vc, x="Nhãn", y="Số lượng",
                        title="Phân phối biến mục tiêu: PD (vỡ nợ)",
                        color="Nhãn",
                    )
                else:
                    vc = df[var].value_counts().sort_index().reset_index()
                    vc.columns = [var, "Số lượng"]
                    fig = px.bar(
                        vc, x=var, y="Số lượng",
                        title=f"Phân phối biến: {var}",
                    )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# THÀNH PHẦN 5: TAB KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH
# ------------------------------------------------------------
with tab3:
    if "model" not in st.session_state:
        st.info("ℹ️ Vui lòng bấm nút '🚀 Huấn luyện & Kiểm định mô hình' ở thanh bên để xem kết quả.")
    else:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        y_proba = st.session_state["y_proba"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        st.subheader("Chỉ tiêu kiểm định tổng quan")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f"Thực tế: {TARGET_LABELS[i]}" for i in sorted(y_test.unique())],
                columns=[f"Dự báo: {TARGET_LABELS[i]}" for i in sorted(y_test.unique())],
            )
            st.dataframe(cm_df, use_container_width=True)

        with col_b:
            st.subheader("Đường cong ROC")
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_fig = px.area(
                    x=fpr, y=tpr,
                    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                    title=f"ROC Curve (AUC = {auc:.3f})",
                )
                roc_fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                roc_fig.update_layout(height=350)
                st.plotly_chart(roc_fig, use_container_width=True)
            except Exception:
                st.warning("Không thể vẽ đường cong ROC với dữ liệu hiện tại.")

        st.subheader("Báo cáo phân loại chi tiết (Classification Report)")
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Bảng kết quả dự báo trên tập kiểm tra")
        with st.container(height=300):
            st.dataframe(st.session_state["results_df"], use_container_width=True)

# ------------------------------------------------------------
# THÀNH PHẦN 6: TAB SỬ DỤNG MÔ HÌNH
# ------------------------------------------------------------
with tab4:
    if "model" not in st.session_state:
        st.info("ℹ️ Vui lòng bấm nút '🚀 Huấn luyện & Kiểm định mô hình' ở thanh bên trước khi sử dụng mô hình.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["feature_cols"]
        train_df = st.session_state["train_data_ref"]

        mode = st.radio(
            "Chọn chế độ dự báo",
            options=["Nhập trực tiếp", "Tải tệp dữ liệu"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp":
            st.caption("Nhập điểm khảo sát (thang 1-5) cho từng tiêu chí, sau đó bấm 'Dự báo'.")
            with st.form("predict_form"):
                input_values = {}
                for group_name, cols in FEATURE_GROUPS.items():
                    st.markdown(f"**{group_name}**")
                    n_cols = min(len(cols), 5)
                    grid = st.columns(n_cols)
                    for i, col_name in enumerate(cols):
                        col_data = train_df[col_name]
                        with grid[i % n_cols]:
                            input_values[col_name] = st.number_input(
                                col_name,
                                min_value=int(col_data.min()),
                                max_value=int(col_data.max()),
                                value=int(col_data.median()),
                                step=1,
                                help=f"Điểm khảo sát cho câu hỏi {col_name} (thang {int(col_data.min())}-{int(col_data.max())}).",
                                key=f"input_{col_name}",
                            )
                submitted = st.form_submit_button("🔮 Dự báo", type="primary", use_container_width=True)

            if submitted:
                try:
                    x_new = pd.DataFrame([input_values])[feature_cols]
                    pred = model.predict(x_new)[0]
                    proba = model.predict_proba(x_new)[0][1]

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if pred == 1:
                            st.error(f"⚠️ Dự báo: **{TARGET_LABELS[1]}**")
                        else:
                            st.success(f"✅ Dự báo: **{TARGET_LABELS[0]}**")
                    with res_col2:
                        st.metric("Xác suất vỡ nợ", f"{proba * 100:.1f}%")
                except Exception as e:
                    st.error(f"❌ Không thể dự báo: {e}")

        else:
            st.caption(
                "Tải lên tệp CSV chứa đúng các cột đầu vào (24 biến): " + ", ".join(feature_cols)
            )
            batch_file = st.file_uploader(
                "Tải tệp CSV để dự báo hàng loạt",
                type=["csv"],
                key="batch_predict_uploader",
                help="Tệp phải chứa đúng các cột biến đầu vào của mô hình (không cần cột PD).",
            )
            if batch_file is not None:
                try:
                    new_df = pd.read_csv(batch_file)
                    missing_batch_cols = [c for c in feature_cols if c not in new_df.columns]
                    if missing_batch_cols:
                        st.error(
                            "❌ Tệp thiếu các cột bắt buộc: " + ", ".join(missing_batch_cols)
                        )
                    else:
                        x_batch = new_df[feature_cols]
                        preds = model.predict(x_batch)
                        probas = model.predict_proba(x_batch)[:, 1]

                        out_df = new_df.copy()
                        out_df["Dự báo (PD)"] = preds
                        out_df["Xác suất vỡ nợ"] = probas

                        st.subheader("Kết quả dự báo")
                        with st.container(height=350):
                            st.dataframe(out_df, use_container_width=True)

                        csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải kết quả dự báo (CSV)",
                            data=csv_bytes,
                            file_name="ket_qua_du_bao_PD.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"❌ Không thể xử lý tệp: {e}")
