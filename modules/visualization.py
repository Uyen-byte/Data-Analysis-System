"""Biểu đồ dùng trong trang phân tích dữ liệu."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_bar_chart(
    df: pd.DataFrame, x: str, y: str, aggregation: str = "mean"
) -> go.Figure:
    """Vẽ biểu đồ cột sau khi gộp giá trị theo từng nhóm."""
    grouped = (
        df.groupby(x, dropna=False, sort=False)[y]
        .agg(aggregation)
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x=x,
        y=y,
        labels={x: x, y: y},
        title=f"{aggregation.title()} của {y} theo {x}",
        color_discrete_sequence=["#2563eb"],
    )
    figure.update_layout(template="plotly_white", margin=dict(t=55, r=20, b=35, l=20))
    return figure


def create_line_chart(
    df: pd.DataFrame, x: str, y: str, aggregation: str = "mean"
) -> go.Figure:
    """Vẽ xu hướng của chỉ số theo cột được chọn."""
    grouped = (
        df.groupby(x, dropna=False, sort=False)[y]
        .agg(aggregation)
        .reset_index()
    )
    if pd.api.types.is_numeric_dtype(grouped[x]):
        grouped = grouped.sort_values(x)

    figure = px.line(
        grouped,
        x=x,
        y=y,
        markers=True,
        labels={x: x, y: y},
        title=f"{aggregation.title()} của {y} theo {x}",
        color_discrete_sequence=["#2563eb"],
    )
    figure.update_layout(template="plotly_white", margin=dict(t=55, r=20, b=35, l=20))
    return figure


def create_pie_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Vẽ tỷ lệ số bản ghi theo từng giá trị phân loại."""
    counts = (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="Số lượng")
    )
    figure = px.pie(
        counts,
        names=column,
        values="Số lượng",
        hole=0.45,
        title=f"Phân bố {column}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    figure.update_layout(template="plotly_white", margin=dict(t=55, r=20, b=20, l=20))
    return figure


def create_scatter_chart(
    df: pd.DataFrame, x: str, y: str, color: str | None = None
) -> go.Figure:
    """Vẽ mối quan hệ giữa hai cột số."""
    figure = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        opacity=0.65,
        labels={x: x, y: y},
        title=f"Mối quan hệ giữa {x} và {y}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    figure.update_layout(template="plotly_white", margin=dict(t=55, r=20, b=35, l=20))
    return figure


def create_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    """Hiển thị ma trận tương quan dưới dạng bản đồ nhiệt."""
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Ma trận tương quan",
    )
    figure.update_layout(template="plotly_white", margin=dict(t=60, r=20, b=35, l=20))
    return figure