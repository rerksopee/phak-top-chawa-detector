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
# CSS (เวอร์ชันดีไซน์ดั้งเดิมของคุณเน่)
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
# ฟังก์ชันตรวจจับและคำนวณพื้นที่อิงจากกรอบทดลอง 1x1 เมตร
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
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
            # 🛠️ สูตรคำนวณพื้นที่อิงจากกรอบ 1x1 เมตร (ชดเชยระยะ Perspective ทัศนียภาพ)
            # -----------------------------------------------------------------
            real_area_m2 = 0.0
            height_pixels = frame.shape[0]  # ความสูงแนวตั้งของรูปภาพ
            
            # วนลูปคำนวณพื้นที่รายพิกเซลตามพิกัดแนวตั้ง (แกน Y)
            for y_pixel in ys:
                # แปลงพิกัดแกน Y ให้อยู่ในรูปสัดส่วนพิกเซล (0.0 บนสุดภาพ = ระยะไกล , 1.0 ล่างสุดภาพ = ระยะใกล้)
                norm_y = y_pixel / height_pixels
                
                # ฟังก์ชันประมาณการจำนวนพิกเซลทั้งหมดของกรอบ 1x1 เมตร ณ พิกัดความลึกแนวตั้งนั้น ๆ
                # สอบเทียบ (Calibrate) จากสัดส่วนภาพถ่ายของคุณ:
                # - ถ้าระยะใกล้ (ด้านล่างภาพ): กรอบ 1x1 เมตรจะมีขนาดพื้นที่ประมาณ 85,000 พิกเซล
                # - ถ้าระยะไกล (ด้านบนภาพ): กรอบ 1x1 เมตรจะหดเล็กลงเหลือพื้นที่ประมาณ 22,000 พิกเซล
                frame_pixels_at_y = 22000.0 + (63000.0 * norm_y)
                
                # ค่าน้ำหนักพื้นที่ของพิกเซลนี้ (ตารางเมตร) = 1 ตร.ม. / จำนวนพิกเซลของกรอบ ณ จุดนั้น
                pixel_area_m2 = 1.0 / frame_pixels_at_y
                real_area_m2 += pixel_area_m2

            # ปัดเศษทศนิยม 4 ตำแหน่ง
            real_area_m2 = round(real_area_m2, 4)

            # ตรวจสอบความถูกต้องทางกายภาพ (เนื่องจากผักตบชวาอยู่ในกรอบ 1x1 ม. พื้นที่ไม่มีทางเกิน 1 ตร.ม.)
            if real_area_m2 > 1.0:
                real_area_m2 = 1.0

            # รูปแบบข้อความรายงานผลหน่วยตารางเมตร (ตร.ม.)
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม. (x={cx}, y={cy})")

            # วาดกรอบและแสดงผลบนรูปภาพ (โค้ดดั้งเดิมของคุณ)
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
# UPLOAD INPUT (รับรูปใบเดียวเหมือนเดิม)
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
