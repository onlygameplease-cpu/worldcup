import streamlit as st

st.set_page_config(page_title="World Cup Quant System", page_icon="🏆", layout="wide")

page_scan = st.Page("worldcup.py", title="App Scan Dữ Liệu", icon="📸")
page_predict = st.Page("worldcup_predict.py", title="Cỗ Máy Dự Đoán 9/10", icon="🧠")

pg = st.navigation([page_scan, page_predict])
pg.run()
