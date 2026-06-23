import streamlit as st

import pandas as pd

import base64

import json

import os

import requests

import time

import io

import numpy as np

import difflib

from PIL import ImageGrab, Image

from datetime import datetime



# Lấy API Key từ biến môi trường hoặc cấu hình

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = "gemini-3.5-pro"  # Sử dụng model thông minh và bắt ngữ cảnh tốt

DB_FILE = "dataworldcup/advanced_stats.csv"



from features import calculate_advanced_features



def extract_stats_from_image(img_bytes: bytes, key: str, model: str) -> dict:

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    

    prompt = '''Bạn là một chuyên gia OCR và Trích xuất Dữ liệu Thể thao cấu trúc cao. Nhiệm vụ của bạn là phân tích ảnh chụp màn hình bảng thống kê trận đấu bóng đá (từ Sofascore/Flashscore) và trích xuất chính xác các chỉ số thành định dạng JSON.



HƯỚNG DẪN TRÍCH XUẤT CHẶT CHẼ:

1. Xác định đúng đội Chủ nhà (Home - bên trái) và Đội khách (Away - bên phải).

2. Tìm kiếm và ưu tiên trích xuất các chỉ số cốt lõi: Ngày thi đấu (Match Date), Bàn thắng (Goals), xG (Số bàn thắng được kỳ vọng), xGOT (xG cú sút trúng đích), xA (Kiến tạo dự kiến), Chạm bóng trong vòng cấm (Touches in Opponent Box), Chuyền bóng 1/3 cuối sân (Passes into Final Third), Sút bị chặn (Blocked Shots), Bàn thắng bị cản phá (Goalkeeper Saves/Prevented Goals), Phạt góc (Corner Kicks), Thẻ vàng (Yellow Cards), Lỗi (Fouls), Tắc bóng (Tackles), Tranh chấp thành công (Duels won), Giải nguy (Clearances), Tạt bóng (Crosses), Tỷ lệ cầm bóng (Ball possession). Ngoài ra, tìm kiếm Xếp hạng FIFA (FIFA Rank) thường nằm ngay dưới tên đội bóng (Ví dụ: "FIFA: 17.").

3. Nếu một chỉ số có dạng tỷ lệ hoặc phân số (Ví dụ: Chuyền bóng 84% (424/503), Tắc bóng 78% (7/9)), chỉ trích xuất phần số lượng tổng (Ví dụ: 503, 9) hoặc số phần trăm theo đúng key yêu cầu dưới đây. Trừ Tỷ lệ cầm bóng (Ball possession) trích xuất con số %. Bỏ dấu %.

4. Nếu chỉ số nào hoàn toàn không xuất hiện trong ảnh, hãy để giá trị là null.

5. CHỈ trả ra chuỗi định dạng STRICT JSON hợp lệ, KHÔNG viết thêm bất kỳ từ giải thích nào ngoài khối JSON (không code block markdown). Bắt đầu bằng '{' và kết thúc bằng '}'.



ĐỊNH DẠNG JSON MẪU BẮT BUỘC:

{

  "HomeTeam": "Tên đội nhà",

  "AwayTeam": "Tên đội khách",

  "Home_Rank": 0,

  "Away_Rank": 0,

  "HomeScore": 0,

  "AwayScore": 0,

  "MatchDate": "Ngày tháng năm thi đấu (VD: 15/06/2024)",

  "Time": "Thời gian (vd: FT, 45+2...)",

  "xG_Home": 0.0,

  "xG_Away": 0.0,

  "xGOT_Home": 0.0,

  "xGOT_Away": 0.0,

  "xA_Home": 0.0,

  "xA_Away": 0.0,

  "Total_Shots_Home": 0,

  "Total_Shots_Away": 0,

  "Touches_In_Box_Home": 0,

  "Touches_In_Box_Away": 0,

  "Passes_Final_Third_Home": 0,

  "Passes_Final_Third_Away": 0,

  "Blocked_Shots_Home": 0,

  "Blocked_Shots_Away": 0,

  "Goals_Prevented_Home": 0.0,

  "Goals_Prevented_Away": 0.0,

  "Corners_Home": 0,

  "Corners_Away": 0,

  "Yellow_Cards_Home": 0,

  "Yellow_Cards_Away": 0,

  "Fouls_Home": 0,

  "Fouls_Away": 0,

  "Tackles_Home": 0,

  "Tackles_Away": 0,

  "Duels_Won_Home": 0,

  "Duels_Won_Away": 0,

  "Clearances_Home": 0,

  "Clearances_Away": 0,

  "Crosses_Home": 0,

  "Crosses_Away": 0,

  "Possession_Home": 0,

  "Possession_Away": 0

}

'''

    payload = {

        "contents": [

            {

                "parts": [

                    {"text": prompt},

                    {"inlineData": {"mimeType": "image/png", "data": img_b64}}

                ]

            }

        ],

        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}

    }

    

    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)

            if resp.status_code in [503, 429]:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return {"error": f"Lỗi gọi Gemini API ({resp.status_code}) sau {max_retries} lần thử: {resp.text}"}
            elif resp.status_code != 200:
                return {"error": f"Lỗi gọi Gemini API ({resp.status_code}): {resp.text}"}
            
            data = resp.json()
            raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            raw_text = raw_text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
                
            parsed_json = json.loads(raw_text)
            return {"parsed": parsed_json, "raw": raw_text}
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": f"Lỗi Exception sau {max_retries} lần thử: {str(e)}"}



st.set_page_config(page_title="World Cup Stats AI", page_icon="🏆", layout="wide")



st.title("🏆 Trợ lý AI - Nhập Liệu World Cup")

st.markdown("Dán (Paste) hoặc Upload ảnh chụp màn hình thống kê trận đấu, AI sẽ tự động đọc các chỉ số (Possession, xG, Shots...) và lưu vào file **dataworldcup/worldcup_stats.csv**.")



# 1. Cấu hình Key

with st.expander("⚙️ Cấu hình API", expanded=not bool(GEMINI_API_KEY)):

    user_key = st.text_input("Nhập GEMINI_API_KEY (nếu chưa có sẵn)", value=GEMINI_API_KEY, type="password")

    selected_model = st.selectbox("Chọn phiên bản AI Model", ["gemini-2.5-flash", "gemini-3.5-flash"], index=1)



if not user_key:

    st.warning("Vui lòng cung cấp GEMINI_API_KEY để sử dụng tính năng OCR của Google AI.")

    st.stop()



st.divider()



col1, col2 = st.columns([1, 1])



# 2. Vùng tải ảnh lên

with col1:

    st.subheader("1. Nạp ảnh trận đấu")

    

    img_bytes = None

    img_display = None

    

    # Nút Paste

    st.info("Cách 1: Chụp/Copy ảnh (Ctrl+C hoặc PrintScreen) rồi bấm nút dưới đây:")

    if st.button("📋 Đọc ảnh từ Clipboard (Nhanh nhất)", use_container_width=True):

        try:

            img = ImageGrab.grabclipboard()

            if img is None:

                st.error("Không tìm thấy ảnh trong Clipboard. Máy đã Copy ảnh chưa?")

            else:

                if isinstance(img, list) and len(img) > 0 and isinstance(img[0], str):

                    img = Image.open(img[0])

                if hasattr(img, "save"):

                    buf = io.BytesIO()

                    img.save(buf, format="PNG")

                    st.session_state["pasted_img"] = buf.getvalue()

                    st.session_state["img_obj"] = img

                else:

                    st.error("Dữ liệu Clipboard không phải là ảnh hợp lệ.")

        except Exception as e:

            st.error(f"Lỗi đọc clipboard: {e}")

            

    # File Uploader

    st.markdown("---")

    st.markdown("Cách 2: Kéo thả file ảnh vào đây:")

    up_img = st.file_uploader("Upload file ảnh", type=["png", "jpg", "jpeg"])

    if up_img is not None:

        st.session_state["pasted_img"] = up_img.getvalue()

        st.session_state["img_obj"] = Image.open(up_img)

        

    # Xử lý OCR

    if "pasted_img" in st.session_state:

        img_bytes = st.session_state["pasted_img"]

        img_display = st.session_state["img_obj"]

        

        

        if st.button("🚀 Cho AI Trích Xuất Dữ Liệu", type="primary", use_container_width=True):

            with st.spinner("AI đang căng mắt đọc chỉ số..."):

                res = extract_stats_from_image(img_bytes, user_key, selected_model)

                if "error" in res:

                    st.error(res["error"])

                else:

                    st.session_state["ocr_result"] = res["parsed"]

                    st.success("Trích xuất thành công! Hãy xem dữ liệu bên phải.")

        st.image(img_display, caption="Ảnh đang chờ xử lý", use_container_width=True)



# 3. Kết quả và Lưu Data

with col2:

    st.subheader("2. Dữ liệu trích xuất")

    

    if "ocr_result" in st.session_state:

        result = st.session_state["ocr_result"]

        

        # Thêm timestamp

        if "Timestamp" not in result:

            result["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            

        df_new = pd.DataFrame([result])

        df_new = calculate_advanced_features(df_new)

        

        st.dataframe(df_new, use_container_width=True)

        

        if st.button("💾 Xác nhận Lưu vào CSV (advanced_stats.csv)", type="primary"):

            os.makedirs("dataworldcup", exist_ok=True)

            

            if os.path.exists(DB_FILE):

                df_existing = pd.read_csv(DB_FILE)

                

                # Xử lý trùng lặp (Deduplication)

                home = df_new.iloc[0]["HomeTeam"]

                away = df_new.iloc[0]["AwayTeam"]
                
                mdate = df_new.iloc[0].get("MatchDate", "")

                if "MatchDate" not in df_existing.columns:
                    df_existing["MatchDate"] = np.nan

                if pd.isna(mdate) or str(mdate).strip() == "":
                    # Nếu không có ngày thi đấu, tạm thời lọc theo cả Score để tránh xóa nhầm trận khác
                    mask = (df_existing["HomeTeam"] == home) & (df_existing["AwayTeam"] == away) & (df_existing["HomeScore"] == df_new.iloc[0]["HomeScore"]) & (df_existing["AwayScore"] == df_new.iloc[0]["AwayScore"])
                else:
                    mask = (df_existing["HomeTeam"] == home) & (df_existing["AwayTeam"] == away) & (df_existing["MatchDate"] == mdate)

                

                if mask.any():

                    # Xóa dòng cũ nếu trùng trận đấu

                    df_existing = df_existing[~mask]

                    st.info(f"Đã ghi đè (Overwrite) trận {home} vs {away} bị trùng lặp trong Database.")

                    

                df_final = pd.concat([df_existing, df_new], ignore_index=True)

            else:

                df_final = df_new

                

            df_final.to_csv(DB_FILE, index=False)

            st.success("Đã lưu vào CSV thành công!")



st.markdown("---")

st.subheader("🗄️ Dữ liệu Database hiện tại (advanced_stats.csv)")

if os.path.exists(DB_FILE):

    df_db = pd.read_csv(DB_FILE)

    st.write(f"Tổng số trận đã quét: **{len(df_db)}**")

    st.dataframe(df_db, use_container_width=True)

else:
    st.info("Database hiện đang trống. Hãy quét ảnh để thêm dữ liệu!")

# --- BACKUP & RESTORE ---
st.markdown("---")
st.subheader("💾 Backup & Đồng bộ dữ liệu (Dành cho Du lịch)")
st.info("Khi chạy trên Web (Streamlit Cloud), dữ liệu quét mới sẽ bị mất nếu server ngủ đông. Hãy tải file CSV về máy/điện thoại, và Upload ngược lại khi cần dùng.")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            st.download_button(
                label="⬇️ Tải file Database (CSV) về máy",
                data=f,
                file_name="advanced_stats.csv",
                mime="text/csv",
                type="primary"
            )

with col2:
    uploaded_db = st.file_uploader("⬆️ Upload file CSV để Phục hồi", type=["csv"], key="restore_db")
    if uploaded_db is not None:
        if st.button("⚠️ Xác nhận Phục hồi (Ghi đè DB hiện tại)"):
            with open(DB_FILE, "wb") as f:
                f.write(uploaded_db.getbuffer())
            st.success("✅ Phục hồi Database thành công! Hãy tải lại trang (F5).")
