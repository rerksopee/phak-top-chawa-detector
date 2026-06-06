import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG & CSS ดีไซน์สีเขียวของคุณ
# =========================
st.set_page_config(page_title="Phak Top Chawa", page_icon="🌿", layout="centered")
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important; }
.stMarkdown p, .stMarkdown span, .stText, .stSubheader, .stHeader, h1, h2, h3 { color: #1b5e20 !important; }
.main-title { background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; padding: 24px; border-radius: 22px; text-align: center; font-size: 42px; font-weight: 700; margin-bottom: 16px; }
.sub-title { text-align: center; color: #1b5e20 !important; font-size: 20px; margin-bottom: 35px; font-weight: 500; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255, 255, 255, 0.25) !important; border: 1px dashed #1b5e20 !important; border-radius: 18px !important; }
.stButton button { width: 100%; background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; border-radius: 16px !important; padding: 12px 28px !important; font-size: 18px !important; font-weight: 700 !important; }
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
# ฟังก์ชันคำนวณพื้นที่ระบบอ้างอิงสเกลกลางมาตรฐาน
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

            # กรองจุดพิกเซลขนาดเล็กมาก ๆ ที่อาจเป็น Noise ออกไป
            if area_pixels < 80:
                continue

            # -----------------------------------------------------------------
            # 🎯 ตรรกะใหม่: Universal Calibration Matrix (เลิกอิงตามแกน Y)
            # -----------------------------------------------------------------
            # ค่าคงที่สเกลเฉลี่ยคำนวณจากกรอบทดลอง 1x1 เมตรของคุณในระยะสายตาทั่วไป
            # เพื่อแปรสัดส่วนจำนวนพิกเซลออกมาเป็นตารางเมตรโดยไม่แกว่งตามตำแหน่งภาพ
            pixel_calibration_constant = 11500.0
            
            # คำนวณพื้นที่ดิบทางคณิตศาสตร์
            real_area_m2 = area_pixels / pixel_calibration_constant

            # ปรับปรุงการคำนวณสำหรับกอผักตบธรรมชาติที่มีขนาดใหญ่มาก (เพื่อไม่ให้ค่าหดตัวผิดความจริง)
            if real_area_m2 > 1.5:
                real_area_m2 = real_area_m2 * 2.3  # ชดเชยระยะลึกของภาพมุมกว้างภายนอก

            real_area_m2 = round(real_area_m2, 2)
            
            if real_area_m2 <= 0:
                real_area_m2 = 0.01

            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม.")

            # ค้นหาพิกัดเพื่อวาดกรอบสี่เหลี่ยมรอบกอผักตบชวา
            contours, _ = cv2.findContours(
                (binary * 255).astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # แสดงผลตัวเลขขนาดพื้นที่บนภาพ
                cv2.putText(
                    frame,
                    f"{i + 1} ({real_area_m2} m2)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    return frame, output_text

# =========================
# UI HEADER / RUN APP
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวามาตรฐานสากล</div>
""", unsafe_allow_html=True)

st.subheader("📤 อัปโหลดรูปภาพ")
uploaded_file = st.file_uploader("รองรับไฟล์ภาพ JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
analyze = st.button("วิเคราะห์พื้นที่")

if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("กำลังคำนวณพื้นที่จริงตามสเกลมาตรฐาน..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการตรวจจับขนาดพื้นที่")
        if texts:
            for t in texts: st.write(t)
        else: st.warning("ไม่พบกอผักตบชวาในภาพ")

        st.image(result_rgb, use_container_width=True)
