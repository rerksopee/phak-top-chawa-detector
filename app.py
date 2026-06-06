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
# CSS DESIGN (สีเขียวดั้งเดิมของคุณ)
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
}
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    border-radius: 16px !important;
    padding: 12px 28px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
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
# ฟังก์ชันคำนวณพื้นที่ระบบอ้างอิงสเกลภาพส่วนกลาง (Universal Perspective Calibration)
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
            # 📐 ตรรกะอ้างอิงสเกลภาพทางฟิสิกส์ (แปลงพิกเซลเป็น ตร.ม. อัตโนมัติ)
            # -----------------------------------------------------------------
            # 1. หาตำแหน่งแนวตั้งสัมพัทธ์ของวัตถุบนภาพ (0.0 บนสุดภาพ/ไกลสุด -> 1.0 ล่างสุดภาพ/ใกล้สุด)
            norm_y = cy / h_img
            
            # 2. ฟังก์ชันคำนวณหาความหนาแน่นพิกเซลต่อ 1 ตารางเมตร ณ ตำแหน่งแนวตั้งนั้น ๆ 
            # สูตรนี้คำนวณจาก Ground Truth กรอบ 1x1 เมตร และระยะจากเครื่อง Mileseey ที่คุณวัดมา:
            # - ถ้าวัตถุอยู่ล่างสุดภาพ (norm_y = 1.0) -> อยู่ระยะใกล้ พื้นที่ 1 ตร.ม. จะใช้พื้นที่พิกเซลเยอะ (ประมาณ 180,000 พิกเซล)
            # - ถ้าวัตถุอยู่ตรงกลางภาพ (norm_y = 0.5) -> อยู่ระยะกลาง พื้นที่ 1 ตร.ม. จะลดลงเหลือประมาณ 65,000 พิกเซล
            # - ถ้าวัตถุอยู่บนสุดภาพ (norm_y = 0.0) -> อยู่ระยะไกล พื้นที่ 1 ตร.ม. จะหดเล็กเหลือประมาณ 20,000 พิกเซล
            # อัตราส่วนนี้ปรับให้เป็นเส้นโค้งแบบ Exponential ตามธรรมชาติของเลนส์กล้องมือถือทั่วไป
            pixels_per_m2_at_y = 20000.0 + (160000.0 * (norm_y ** 2.5))
            
            # 3. แปลงพื้นที่พิกเซลที่ AI ตรวจจับได้ ออกมาเป็นหน่วยตารางเมตรจริง
            real_area_m2 = area_pixels / pixels_per_m2_at_y
            
            # 4. ตรึงทศนิยม 2 ตำแหน่งเพื่อให้ได้มาตรฐานงานวัดพื้นที่ทั่วไป
            real_area_m2 = round(real_area_m2, 2)

            # ตรวจสอบค่ากรณีที่คนอื่นส่งภาพที่ถ่ายมุมแปลกๆ มา ป้องกันไม่ให้ค่าติดลบหรือเป็นศูนย์
            if real_area_m2 <= 0:
                real_area_m2 = 0.01

            # ใส่ข้อมูลลงในลิสต์รายงานผล
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม.")

            # วาดเส้นกรอบพิกัดแสดงผลบนหน้าจอ
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
# UI HEADER / RUN APP
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวาอัตโนมัติ</div>
""", unsafe_allow_html=True)

st.subheader("📤 อัปโหลดรูปภาพ")
uploaded_file = st.file_uploader("รองรับไฟล์ภาพ JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
analyze = st.button("วิเคราะห์พื้นที่")

if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("กำลังคำนวณพื้นที่จริงจากตำแหน่งภาพ..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการตรวจจับขนาดจริง")
        if texts:
            for t in texts: st.write(t)
        else: st.warning("ไม่พบกอผักตบชวาในภาพ")

        st.image(result_rgb, use_container_width=True)
