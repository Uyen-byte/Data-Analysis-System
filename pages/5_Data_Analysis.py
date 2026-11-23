"""Dashboard thống kê mô tả, tương quan và trực quan hóa dữ liệu."""

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_analysis import DataAnalyzer, is_identifier_column
from modules.visualization import (
    create_bar_chart,
    create_correlation_heatmap,
    create_line_chart,
    create_pie_chart,
    create_scatter_chart,
)


st.set_page_config(
    page_title="Phân tích dữ liệu",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 12px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] {color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"


@st.cache_data(show_spinner="Đang đọc dữ liệu...")
def load_data(source: str | bytes) -> pd.DataFrame:
    """Đọc dữ liệu mặc định hoặc dữ liệu CSV người dùng tải lên."""
    if isinstance(source, bytes):
        return pd.read_csv(BytesIO(source))
    return pd.read_csv(source)


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


st.title("Phân tích dữ liệu")
st.caption("Thống kê mô tả, khám phá tương quan và trực quan hóa dữ liệu của bạn.")

with st.sidebar:
    st.header("Nguồn dữ liệu")
    uploaded_file = st.file_uploader(
        "Tải tệp CSV",
        type=["csv"],
        key="data_analysis_upload",
    )

df = None
if uploaded_file is not None:
    data_source = uploaded_file.getvalue()
    try:
        df = load_data(data_source)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.error(f"Không thể đọc tệp CSV: {error}")
        st.stop()
    st.session_state["analysis_uploaded_dataset"] = df.copy()
    st.session_state["analysis_uploaded_file_name"] = uploaded_file.name
    source_label = f"Đang dùng tệp: `{uploaded_file.name}`"
else:
    analysis_dataset = st.session_state.get("analysis_uploaded_dataset")
    cleaned_dataset = st.session_state.get("cleaned_dataset")
    session_dataset = st.session_state.get("dataset")
    uploaded_dataset = st.session_state.get("uploaded_dataset")

    if isinstance(analysis_dataset, pd.DataFrame) and not analysis_dataset.empty:
        df = analysis_dataset.copy()
        file_name = st.session_state.get("analysis_uploaded_file_name", "CSV")
        source_label = f"Đang giữ tệp đã tải: `{file_name}`"
    elif isinstance(cleaned_dataset, pd.DataFrame) and not cleaned_dataset.empty:
        df = cleaned_dataset.copy()
        source_label = "Đang dùng dữ liệu đã làm sạch."
    elif isinstance(session_dataset, pd.DataFrame) and not session_dataset.empty:
        df = session_dataset.copy()
        source_label = "Đang dùng dữ liệu trong phiên làm việc."
    elif isinstance(uploaded_dataset, pd.DataFrame) and not uploaded_dataset.empty:
        df = uploaded_dataset.copy()
        source_label = "Đang dùng dữ liệu đã tải lên."
    else:
        source_label = f"Dữ liệu mặc định: `{DEFAULT_DATA_PATH.name}`"

with st.sidebar:
    st.caption(source_label)

try:
    if df is None:
        df = load_data(str(DEFAULT_DATA_PATH))
except (
    FileNotFoundError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    UnicodeDecodeError,
) as error:
    st.error(f"Không thể đọc tệp CSV: {error}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("Tệp CSV không có dữ liệu để phân tích.")
    st.stop()

analyzer = DataAnalyzer(df)
overview = analyzer.get_overview()
numeric_df = analyzer.get_numeric_data()
numeric_columns = numeric_df.columns.tolist()
plot_df = df.copy()
for column in numeric_columns:
    plot_df[column] = numeric_df[column]

categorical_columns = [
    column
    for column in df.select_dtypes(include=["object", "category", "bool"]).columns
    if column not in numeric_columns
    and not is_identifier_column(column)
    and df[column].notna().any()
]
bar_group_columns = [
    column for column in categorical_columns
    if df[column].nunique(dropna=False) <= 50
]
pie_columns = [
    column for column in categorical_columns
    if df[column].nunique(dropna=False) <= 12
]

with st.sidebar:
    st.divider()
    if numeric_columns:
        selected_metric = st.selectbox(
            "Cột số liệu",
            numeric_columns,
            index=default_index(numeric_columns, "overall_score"),
        )
    else:
        selected_metric = None
        st.info("Tệp này không có cột số để tính các chỉ số số học.")

overview_tab, charts_tab, correlation_tab, data_tab = st.tabs(
    ["Tổng quan", "Biểu đồ", "Tương quan", "Dữ liệu"]
)

with overview_tab:
    st.subheader("Tổng quan dữ liệu")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Số dòng", f"{overview['total_rows']:,}")
    metric_columns[1].metric("Số cột", f"{overview['total_columns']:,}")
    metric_columns[2].metric("Giá trị thiếu", f"{overview['missing_values']:,}")
    metric_columns[3].metric("Dòng trùng lặp", f"{overview['duplicate_rows']:,}")

    if selected_metric is not None:
        st.subheader(f"Chỉ số của cột: {selected_metric}")
        column_stats = analyzer.analyze_column(selected_metric)
        stat_columns = st.columns(4)
        stat_columns[0].metric("Trung bình", f"{column_stats['Mean']:,.2f}")
        stat_columns[1].metric("Tổng", f"{column_stats['Sum']:,.2f}")
        stat_columns[2].metric("Lớn nhất", f"{column_stats['Max']:,.2f}")
        stat_columns[3].metric("Nhỏ nhất", f"{column_stats['Min']:,.2f}")

    st.subheader("Thống kê mô tả")
    descriptive_statistics = analyzer.get_descriptive_statistics()
    if descriptive_statistics.empty:
        st.info("Không có cột số để tạo thống kê mô tả.")
    else:
        st.dataframe(
            descriptive_statistics.style.format("{:,.2f}"),
            use_container_width=True,
        )

with charts_tab:
    st.subheader("Trực quan hóa")
    aggregation_label = st.selectbox(
        "Cách tổng hợp cho biểu đồ cột và đường",
        ["Trung bình", "Tổng"],
        key="chart_aggregation",
    )
    aggregation = "mean" if aggregation_label == "Trung bình" else "sum"

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("#### Biểu đồ đường")
        if not numeric_columns:
            st.info("Biểu đồ đường cần ít nhất một cột số.")
        else:
            line_x_options = list(dict.fromkeys(numeric_columns + bar_group_columns))
            line_x = st.selectbox(
                "Trục X (đường)",
                line_x_options,
                index=default_index(line_x_options, "study_hours"),
            )
            line_y = st.selectbox(
                "Chỉ số (đường)",
                numeric_columns,
                index=default_index(numeric_columns, "overall_score"),
            )
            st.plotly_chart(
                create_line_chart(plot_df, line_x, line_y, aggregation),
                use_container_width=True,
            )

    with top_right:
        st.markdown("#### Biểu đồ cột")
        if not numeric_columns or not bar_group_columns:
            st.info("Biểu đồ cột cần cột số và cột phân loại.")
        else:
            bar_group = st.selectbox(
                "Nhóm theo (cột)",
                bar_group_columns,
                index=default_index(bar_group_columns, "school_type"),
            )
            bar_value = st.selectbox(
                "Chỉ số (cột)",
                numeric_columns,
                index=default_index(numeric_columns, "overall_score"),
            )
            st.plotly_chart(
                create_bar_chart(plot_df, bar_group, bar_value, aggregation),
                use_container_width=True,
            )

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("#### Biểu đồ tròn")
        if not pie_columns:
            st.info("Biểu đồ tròn cần cột phân loại có tối đa 12 giá trị khác nhau.")
        else:
            pie_column = st.selectbox(
                "Cột phân loại (tròn)",
                pie_columns,
                index=default_index(pie_columns, "school_type"),
            )
            st.plotly_chart(
                create_pie_chart(plot_df, pie_column),
                use_container_width=True,
            )

    with bottom_right:
        st.markdown("#### Biểu đồ phân tán")
        if len(numeric_columns) < 2:
            st.info("Biểu đồ phân tán cần ít nhất hai cột số.")
        else:
            scatter_x_column, scatter_y_column = st.columns(2)
            with scatter_x_column:
                scatter_x = st.selectbox(
                    "Trục X (phân tán)",
                    numeric_columns,
                    index=default_index(numeric_columns, "study_hours"),
                )
            with scatter_y_column:
                scatter_y_options = [
                    column for column in numeric_columns
                    if column != scatter_x
                ]
                scatter_y = st.selectbox(
                    "Trục Y (phân tán)",
                    scatter_y_options,
                    index=default_index(scatter_y_options, "overall_score"),
                )
            color_options = ["Không tô màu"] + bar_group_columns
            color_by = st.selectbox("Phân nhóm màu", color_options)
            st.plotly_chart(
                create_scatter_chart(
                    plot_df,
                    scatter_x,
                    scatter_y,
                    None if color_by == "Không tô màu" else color_by,
                ),
                use_container_width=True,
            )

with correlation_tab:
    st.subheader("Phân tích tương quan")
    correlation = analyzer.get_correlation()
    if correlation.shape[1] < 2:
        st.info("Cần ít nhất hai cột số để phân tích tương quan.")
    else:
        st.plotly_chart(
            create_correlation_heatmap(correlation),
            use_container_width=True,
        )
        st.caption(
            "Hệ số tương quan nằm trong khoảng -1 đến 1. "
            "Giá trị gần 1 là tương quan thuận; gần -1 là tương quan nghịch."
        )

with data_tab:
    st.subheader("Bảng dữ liệu")
    st.caption(f"Hiển thị {len(df):,} dòng và {len(df.columns):,} cột.")
    st.dataframe(df, use_container_width=True, height=520)
    st.download_button(
        "Tải dữ liệu CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_phan_tich.csv",
        mime="text/csv",
    )