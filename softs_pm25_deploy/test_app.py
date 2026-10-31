import streamlit as st

st.set_page_config(page_title="Streamlit Test", layout="wide")

st.title("Hanoi Air Intelligence")
st.success("Streamlit đang hoạt động bình thường.")
st.write("Đây là trang kiểm tra độc lập.")

if st.button("Kiểm tra nút"):
    st.write("Nút hoạt động.")
