import streamlit as st
import pandas as pd
import io

from modules.data_cleaning import DataCleaning


# Tiêu đề trang
st.title("Data Cleaning")
st.caption("Kiểm tra và tiền xử lý dữ liệu")

# Kiểm tra dữ liệu đầu vào
if "dataset" not in st.session_state:
    st.warning("Vui lòng tải dữ liệu ở trang Upload Data trước.")
    st.stop()

df = st.session_state["dataset"]

# Kiểm tra kiểu dữ liệu
if not isinstance(df, pd.DataFrame):
    st.error("Dữ liệu đầu vào không hợp lệ.")
    st.stop()


# Khởi tạo đối tượng DataCleaning
# Chỉ khởi tạo lại khi người dùng tải tập dữ liệu mới
if (
    "data_cleaner" not in st.session_state
    or st.session_state.get("_cleaning_source") is not df
):
    st.session_state["data_cleaner"] = DataCleaning(df)
    st.session_state["_cleaning_source"] = df

# Lấy đối tượng DataCleaning
cleaner = st.session_state["data_cleaner"]

st.subheader("Tổng quan dữ liệu")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tổng số dòng", len(cleaner.df))

col2.metric("Tổng số cột", len(cleaner.df.columns))

col3.metric(
    "Giá trị thiếu",
    int(cleaner.df.isnull().sum().sum())
)

col4.metric(
    "Dòng trùng lặp",
    int(cleaner.df.duplicated().sum())
)

st.subheader("Dữ liệu hiện tại")

st.dataframe(
    cleaner.df,
    use_container_width=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dữ liệu thiếu",
    "Dữ liệu trùng",
    "Chuyển đổi kiểu",
    "Chuẩn hóa",
    "Ngoại lệ",
    "Kết quả"
])

with tab1:

    st.subheader("Kiểm tra dữ liệu thiếu")

    # Hiển thị thống kê dữ liệu thiếu
    missing_result = cleaner.check_missing()

    st.dataframe(
        missing_result,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Xử lý giá trị thiếu")

    strategy_label = st.selectbox(
        "Chọn phương pháp xử lý",
        [
            "Trung bình",
            "Trung vị",
            "Giá trị phổ biến nhất",
            "Giá trị tự nhập"
        ]
    )

    strategy_map = {
        "Trung bình": "mean",
        "Trung vị": "median",
        "Giá trị phổ biến nhất": "mode",
        "Giá trị tự nhập": "value"
    }

    strategy = strategy_map[strategy_label]

    if strategy in ["mean", "median"]:
        available_columns = [
            col for col in cleaner.df.columns
            if pd.api.types.is_numeric_dtype(cleaner.df[col])
        ]

    elif strategy == "mode":
        available_columns = list(cleaner.df.columns)

    else:
        available_columns = list(cleaner.df.columns)

    if strategy == "value":

        if available_columns:

            selected_column = st.selectbox(
                "Chọn cột cần xử lý",
                available_columns
            )

            input_value = st.text_input(
                "Nhập giá trị thay thế"
            )

            if st.button("Thực hiện thay thế", key="fill_custom"):

                try:
                    if input_value.strip() == "":
                        st.warning("Vui lòng nhập giá trị.")
                    else:
                        dtype = cleaner.df[selected_column].dtype

                        if pd.api.types.is_numeric_dtype(dtype):
                            value = pd.to_numeric(input_value)

                        elif pd.api.types.is_datetime64_any_dtype(dtype):
                            value = pd.to_datetime(input_value)

                        else:
                            value = input_value

                        result = cleaner.fill_missing(
                            strategy="value",
                            columns=[selected_column],
                            value=value
                        )

                        st.success(
                            f"Đã xử lý: {result}"
                        )

                        st.dataframe(cleaner.df, use_container_width=True)

                except (ValueError, TypeError) as e:
                    st.error(str(e))

    else:

        if available_columns:

            selected_columns = st.multiselect(
                "Chọn cột cần xử lý",
                available_columns,
                default=available_columns
            )

            if st.button("Thực hiện xử lý", key="fill_missing"):

                if not selected_columns:
                    st.warning("Vui lòng chọn ít nhất một cột.")

                else:
                    try:
                        result = cleaner.fill_missing(
                            strategy=strategy,
                            columns=selected_columns
                        )

                        st.success(
                            f"Kết quả xử lý: {result}"
                        )

                        st.dataframe(
                            cleaner.df,
                            use_container_width=True
                        )

                    except (ValueError, TypeError) as e:
                        st.error(str(e))

        else:
            st.info("Không có cột phù hợp để xử lý.")

    st.divider()

    st.subheader("Xóa dòng chứa dữ liệu thiếu")

    how = st.selectbox(
        "Điều kiện xóa",
        ["any", "all"],
        format_func=lambda x: (
            "Xóa nếu có ít nhất một ô thiếu"
            if x == "any"
            else "Chỉ xóa nếu toàn bộ dòng bị thiếu"
        )
    )

    missing_columns = st.multiselect(
        "Chọn cột kiểm tra (để trống là toàn bộ cột)",
        list(cleaner.df.columns),
        key="remove_missing_columns"
    )

    if st.button("Xóa dữ liệu thiếu", key="remove_missing"):

        try:
            removed = cleaner.remove_missing(
                how=how,
                subset=missing_columns or None
            )

            st.success(f"Đã xóa {removed} dòng dữ liệu thiếu.")

            st.dataframe(
                cleaner.df,
                use_container_width=True
            )

        except (ValueError, KeyError) as e:
            st.error(str(e))

with tab2:

    st.subheader("Kiểm tra dữ liệu trùng lặp")

    duplicate_columns = st.multiselect(
        "Chọn cột dùng để kiểm tra trùng (để trống là toàn bộ cột)",
        list(cleaner.df.columns),
        key="duplicate_columns"
    )

    subset = duplicate_columns or None

    duplicate_result = cleaner.check_duplicates(
        subset=subset
    )

    st.write(
        f"Số dòng trùng lặp: {len(duplicate_result)}"
    )

    st.dataframe(
        duplicate_result,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Xóa dữ liệu trùng lặp", key="delete_duplicates"):

        try:
            removed = cleaner.remove_duplicates(
                subset=subset
            )

            st.success(
                f"Đã xóa {removed} dòng trùng lặp."
            )

            st.dataframe(
                cleaner.df,
                use_container_width=True
            )

        except KeyError as e:
            st.error(str(e))

with tab3:

    st.subheader("Chuyển đổi kiểu dữ liệu")

    # Hiển thị kiểu dữ liệu hiện tại
    st.dataframe(
        cleaner.get_data_types(),
        use_container_width=True,
        hide_index=True
    )

    selected_column = st.selectbox(
        "Chọn cột cần chuyển đổi",
        list(cleaner.df.columns)
    )

    target_type = st.selectbox(
        "Chọn kiểu dữ liệu mới",
        ["integer", "float", "string", "datetime"]
    )

    if st.button("Chuyển đổi kiểu dữ liệu"):

        try:
            result = cleaner.convert_data_types(
                selected_column,
                target_type
            )

            st.success("Đã thực hiện chuyển đổi.")

            st.write("Chi tiết kết quả:", result)

            st.dataframe(
                cleaner.get_data_types(),
                use_container_width=True
            )

        except (ValueError, TypeError, KeyError) as e:
            st.error(str(e))

with tab4:

    st.subheader("Chuẩn hóa dữ liệu")

    text_columns = [
        col for col in cleaner.df.columns
        if cleaner.df[col].dropna().map(
            lambda x: isinstance(x, str)
        ).any()
    ]

    if text_columns:

        selected_columns = st.multiselect(
            "Chọn cột cần chuẩn hóa",
            text_columns,
            default=text_columns
        )

        case_label = st.selectbox(
            "Chọn phương pháp chuẩn hóa",
            [
                "Chỉ loại bỏ khoảng trắng",
                "Chuyển thành chữ thường",
                "Chuyển thành chữ hoa",
                "Viết hoa chữ cái đầu"
            ]
        )

        case_map = {
            "Chỉ loại bỏ khoảng trắng": None,
            "Chuyển thành chữ thường": "lower",
            "Chuyển thành chữ hoa": "upper",
            "Viết hoa chữ cái đầu": "title"
        }

        if st.button("Thực hiện chuẩn hóa"):

            try:
                if not selected_columns:
                    st.warning("Vui lòng chọn ít nhất một cột.")

                else:
                    result = cleaner.standardize_data(
                        columns=selected_columns,
                        case=case_map[case_label]
                    )

                    st.success("Đã chuẩn hóa dữ liệu.")

                    st.write("Kết quả:", result)

                    st.dataframe(
                        cleaner.df,
                        use_container_width=True
                    )

            except (ValueError, KeyError) as e:
                st.error(str(e))

    else:
        st.info("Không có cột dữ liệu văn bản để chuẩn hóa.")

with tab5:

    st.subheader("Phát hiện dữ liệu ngoại lệ")

    numeric_columns = [
        col for col in cleaner.df.columns
        if (
            pd.api.types.is_numeric_dtype(cleaner.df[col])
            and not pd.api.types.is_bool_dtype(cleaner.df[col])
        )
    ]

    if numeric_columns:

        selected_columns = st.multiselect(
            "Chọn cột số cần kiểm tra",
            numeric_columns,
            default=numeric_columns
        )

        factor = st.number_input(
            "Hệ số IQR",
            min_value=0.1,
            value=1.5,
            step=0.1
        )

        if st.button("Phát hiện ngoại lệ"):

            try:
                if not selected_columns:
                    st.warning("Vui lòng chọn cột cần kiểm tra.")

                else:
                    outliers = cleaner.detect_outliers(
                        columns=selected_columns,
                        factor=factor
                    )

                    if outliers.empty:
                        st.success("Không phát hiện dữ liệu ngoại lệ.")
                    else:
                        st.warning(
                            f"Phát hiện {len(outliers)} giá trị ngoại lệ."
                        )

                    st.dataframe(
                        outliers,
                        use_container_width=True,
                        hide_index=True
                    )

            except (ValueError, TypeError, KeyError) as e:
                st.error(str(e))

        st.divider()

        st.subheader("Xử lý dữ liệu ngoại lệ")

        action_label = st.selectbox(
            "Chọn phương pháp xử lý",
            [
                "Điều chỉnh giá trị (Clip)",
                "Xóa dòng chứa ngoại lệ (Remove)"
            ]
        )

        action_map = {
            "Điều chỉnh giá trị (Clip)": "clip",
            "Xóa dòng chứa ngoại lệ (Remove)": "remove"
        }

        if st.button("Thực hiện xử lý ngoại lệ"):

            try:
                if not selected_columns:
                    st.warning("Vui lòng chọn cột cần xử lý.")

                else:
                    result = cleaner.handle_outliers(
                        columns=selected_columns,
                        factor=factor,
                        action=action_map[action_label]
                    )

                    st.success(
                        f"Đã xử lý {result['affected_values']} giá trị ngoại lệ."
                    )

                    st.write("Chi tiết:", result)

                    st.dataframe(
                        cleaner.df,
                        use_container_width=True
                    )

            except (ValueError, TypeError, KeyError) as e:
                st.error(str(e))

    else:
        st.info("Dữ liệu hiện tại không có cột số để kiểm tra ngoại lệ.")

with tab6:

    st.subheader("So sánh dữ liệu trước và sau xử lý")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dữ liệu ban đầu")

        st.dataframe(
            cleaner.original_df,
            use_container_width=True
        )

    with col2:
        st.markdown("### Dữ liệu đã làm sạch")

        cleaned_df = cleaner.get_cleaned_data()

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )

    st.divider()

    st.subheader("Tổng kết quá trình làm sạch")

    summary = cleaner.get_cleaning_summary()

    st.write(
        "Số dòng ban đầu:",
        summary["original_rows"]
    )

    st.write(
        "Số dòng hiện tại:",
        summary["current_rows"]
    )

    st.write(
        "Số cột:",
        summary["columns"]
    )

    st.write(
        "Số giá trị thiếu còn lại:",
        summary["missing_values"]
    )

    st.write(
        "Số dòng trùng lặp còn lại:",
        summary["duplicate_rows"]
    )

    st.write("Lịch sử xử lý:")

    for history in summary["cleaning_history"]:
        st.write("- ", history)

    st.divider()

    st.subheader("Tải dữ liệu đã làm sạch")

    # Xuất dữ liệu CSV
    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Tải dữ liệu CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # Xuất dữ liệu Excel
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        cleaned_df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    st.download_button(
        label="Tải dữ liệu Excel",
        data=excel_buffer.getvalue(),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    # Khôi phục dữ liệu
    if st.button("Khôi phục dữ liệu ban đầu"):

        cleaner.reset_data()

        st.session_state["cleaned_dataset"] = (
            cleaner.get_cleaned_data()
        )

        st.success("Đã khôi phục dữ liệu ban đầu.")

        st.rerun()

# Lưu dữ liệu đã làm sạch vào hệ thống
st.session_state["cleaned_dataset"] = (
    cleaner.get_cleaned_data()
)