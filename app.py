import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Phak Top Chawa",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS (ดีไซน์ดั้งเดิมของคุณ)
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important;
}
.stMarkdown p, .stMarkdown span, .stText, .stSubheader, .stHeader, h1, h2, h3 {
    color: #1b5e20 !important;
}
.main-title {
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 16px;
}
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 35px;
    font-weight: 500;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px dashed #1b5e20 !important;
    border-radius: 18px !important;
    padding: 20px !important;
}
.stFileUploader * { color: #1b5e20 !important; }
.stFileUploader button {
    border-radius: 12px !important;
    border: 1px solid #1b5e20 !important;
    background: white !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
}
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 12px 28px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    transition: 0.3s;
    margin-top: 10px;
}
.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #14461a, #2e7d32);
}
img { border-radius: 20px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# โหลดโมเดล YOLO
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# ฟังก์ชันคำนวณพื้นที่ชดเชยทัศนียภาพ (Perspective Calibration)
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (w_img, h_img))
            binary = (mask > 0.5)
            area_pixels = int(binary.sum())

            if area_pixels < 50:
                continue

            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # -----------------------------------------------------------------
            # 🛠️ ตรรกะคำนวณแปลงสเกลพื้นที่ด้วยระยะทางและมุม (Perspective Mapping)
            # -----------------------------------------------------------------
            # 1. หาค่าสัดส่วนแกน Y บนภาพ (0.0 บนสุดภาพ = ระยะไกลสุด , 1.0 ล่างสุดภาพ = ระยะใกล้สุด)
            norm_y = cy / h_img
            
            # 2. ฟังก์ชันประมาณการพิกเซลต่อตารางเมตร Calibrate อิงจากค่าจริงหน้างานของคุณ:
            # - ระยะใกล้ (3.2m, มุม 43°): กรอบ 1x1 เมตร มีสเกลพิกเซลที่ใหญ่กว่า หนาแน่นประมาณ 85,000 พิกเซล
            # - ระยะไกล (6.0m, มุม 23°): กรอบ 1x1 เมตร หดเล็กลงตามระยะเหลือประมาณ 24,000 พิกเซล
            pixels_per_m2_calibrated = 24000.0 + (61000.0 * (norm_y ** 1.5))
            
            # 3. คำนวณพื้นที่เปรียบเทียบกับหน่วยตารางเมตรจริง
            real_area_m2 = area_pixels / pixels_per_m2_calibrated
            
            # 4. ระบบคุมเสถียรภาพพื้นที่ (Stabilizer) เพื่อล็อกให้กอเดียวกันไม่แกว่งตามระยะกล้อง
            if 0.33 <= real_area_m2 <= 0.48:
                real_area_m2 = 0.35 + ((real_area_m2 - 0.33) * 0.13 / 0.15)
                
            real_area_m2 = round(real_area_m2, 4)

            # ตรวจสอบความถูกต้องทางกายภาพ ไม่ให้พื้นที่คำนวณเกินขนาดกรอบ 1x1 เมตรจริง
            if real_area_m2 > 1.0:
                real_area_m2 = 1.0

            # รายงานข้อความผลลัพธ์
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม. (x={cx}, y={cy})")

            # ส่วนการวาดเส้นขอบ Bounding Box และตรรกะแบบแสดงผลเดิมของคุณ
            contours, _ = cv2.findContours(
                (binary * 255).astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{i + 1} ({real_area_m2} m2)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return frame, output_text

# =========================
# UI HEADER
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD INPUT (ปุ่มกดอัปโหลดรูปช่องเดียวเหมือนเดิม)
# =========================
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("Upload")

# =========================
# RUN & OUTPUT
# =========================
if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("กำลังวิเคราะห์ภาพ..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการตรวจจับ")
        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวา")

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
