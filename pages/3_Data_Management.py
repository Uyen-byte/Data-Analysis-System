import pandas as pd
import streamlit as st

from database.dataset_repository import DatasetRepository
from database.db_connection import connect_db

st.set_page_config(page_title="Data Management", page_icon="🗂️", layout="wide")
st.title("🗂️ Trang Quản Lý Dữ Liệu (Data Management)")
st.write("Xem, tìm kiếm, lọc, sắp xếp, xem thông tin cột và quản lý dữ liệu sinh viên trong PostgreSQL.")

repository = DatasetRepository()

try:
    total_records = repository.count_records()
    st.metric("Tổng số bản ghi trong cơ sở dữ liệu", f"{total_records:,}")

    if total_records == 0:
        st.info("Chưa có dữ liệu trong hệ thống. Hãy tải lên ở trang Upload Data trước nhé.")
    else:
        # --- TÍNH NĂNG 4: Tìm kiếm dữ liệu ---
        keyword = st.text_input(
            "🔍 Tìm kiếm dữ liệu",
            placeholder="Nhập mã sinh viên, giới tính, loại trường, phương pháp học hoặc xếp loại...",
        )

        if keyword.strip():
            rows, columns = repository.search(keyword.strip(), limit=2000)
        else:
            rows, columns = repository.get_all(limit=2000, offset=0)

        if rows:
            dataframe = pd.DataFrame(rows, columns=columns)
            
            # --- TÍNH NĂNG 5: Lọc dữ liệu theo điều kiện ---
            with st.expander("⚙️ Bộ lọc nâng cao (Lọc dữ liệu theo điều kiện)"):
                col_f1, col_f2, col_f3 = st.columns(3)
                
                selected_gender = []
                selected_school = []
                selected_grade = []
                
                if "gender" in dataframe.columns:
                    genders = dataframe["gender"].unique().tolist()
                    selected_gender = col_f1.multiselect("Giới tính (Gender)", genders, default=genders)
                
                if "school_type" in dataframe.columns:
                    schools = dataframe["school_type"].unique().tolist()
                    selected_school = col_f2.multiselect("Loại trường (School Type)", schools, default=schools)
                    
                if "final_grade" in dataframe.columns:
                    grades = dataframe["final_grade"].unique().tolist()
                    selected_grade = col_f3.multiselect("Xếp loại (Final Grade)", grades, default=grades)
                
                # Áp dụng bộ lọc vào DataFrame
                if selected_gender:
                    dataframe = dataframe[dataframe["gender"].isin(selected_gender)]
                if selected_school:
                    dataframe = dataframe[dataframe["school_type"].isin(selected_school)]
                if selected_grade:
                    dataframe = dataframe[dataframe["final_grade"].isin(selected_grade)]

            # --- TÍNH NĂNG 6: Sắp xếp dữ liệu tăng dần hoặc giảm dần ---
            with st.expander("🔄 Sắp xếp dữ liệu"):
                col_s1, col_s2 = st.columns(2)
                sort_col = col_s1.selectbox("Chọn cột để sắp xếp", options=dataframe.columns.tolist(), index=0)
                sort_order = col_s2.radio("Thứ tự sắp xếp", options=["Tăng dần (Ascending)", "Giảm dần (Descending)"], horizontal=True)
                
                ascending_bool = True if "Tăng" in sort_order else False
                dataframe = dataframe.sort_values(by=sort_col, ascending=ascending_bool)

            # --- TÍNH NĂNG 7: Xem thông tin các cột dữ liệu (Column Info) ---
            with st.expander("📊 Xem thông tin các cột dữ liệu"):
                info_df = pd.DataFrame({
                    "Tên cột": dataframe.columns,
                    "Kiểu dữ liệu": dataframe.dtypes.astype(str),
                    "Số lượng giá trị Null": dataframe.isnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True, hide_index=True)

            # Cập nhật trạng thái vào session_state để các module phân tích/tiền xử lý kế tiếp sử dụng
            st.session_state["dataset"] = dataframe.copy()

            # --- TÍNH NĂNG 3 & 9: Xem dữ liệu dưới dạng bảng & xem danh sách đã lưu ---
            st.subheader(f"📋 Danh sách dữ liệu hiển thị ({len(dataframe):,} bản ghi)")
            st.dataframe(dataframe, use_container_width=True, hide_index=True)
            st.caption("Hiển thị tối đa 2.000 bản ghi cho mỗi lần truy vấn.")

            # --- TÍNH NĂNG 10: Xóa bộ dữ liệu đã lưu trong Database ---
            st.markdown("---")
            with st.expander("⚠️ Khu vực quản trị: Xóa dữ liệu"):
                st.warning("Hành động này sẽ xóa sạch toàn bộ dữ liệu trong bảng `student_performance` trên PostgreSQL!")
                confirm_delete = st.checkbox("Tôi xác nhận muốn xóa toàn bộ dữ liệu hệ thống")
                if st.button("Xóa toàn bộ dữ liệu", type="secondary"):
                    if confirm_delete:
                        try:
                            conn = connect_db()
                            with conn.cursor() as cur:
                                cur.execute("TRUNCATE TABLE student_performance;")
                            conn.commit()
                            conn.close()
                            st.success("Đã xóa toàn bộ dữ liệu trong cơ sở dữ liệu thành công!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi khi xóa dữ liệu: {e}")
                    else:
                        st.error("Vui lòng tích chọn ô xác nhận trước khi thực hiện xóa!")

        else:
            st.info("Không tìm thấy dữ liệu phù hợp với điều kiện.")

    if st.button("Làm mới trang"):
        st.rerun()

except Exception as error:
    st.error(f"Không thể tải dữ liệu: {error}")
    st.info("Hãy kiểm tra kết nối PostgreSQL và đảm bảo bảng student_performance đã được tạo bằng trang Upload Data.")