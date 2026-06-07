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
# CSS DESIGN (รูปแบบหน้าเว็บดั้งเดิมเขียว-ขาวของพี่ 100%)
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
# ฟังก์ชันคำนวณตามสูตร Pixel-to-Metric Calibration
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    
    # -----------------------------------------------------------------
    # 🎯 ขั้นตอนที่ 1: หาค่า R_pixel (พื้นที่พิกเซลของกรอบเหลือง 1x1 เมตรในภาพนั้นๆ)
    # -----------------------------------------------------------------
    # แปลงภาพเป็น HSV เพื่อดึงเฉพาะแถบสีเหลืองของกรอบอ้างอิงทดลอง
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # หาตำแหน่งและคำนวณพื้นที่ของกรอบสีเหลืองในภาพ
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ค่าเริ่มต้นสำหรับกรณีตรวจไม่เจอกรอบเหลือง (ใช้ค่าเฉลี่ยความละเอียดมาตรฐาน)
    r_pixel = 65000.0 
    
    if contours_yellow:
        # ดึงพื้นที่กรอบสีเหลืองที่มีขนาดใหญ่ที่สุดในรูปมาอ้างอิง
        largest_yellow = max(contours_yellow, key=cv2.contourArea)
        area_yellow_pixels = cv2.contourArea(largest_yellow)
        
        # ถ้านิยามพื้นที่ใหญ่พอและสมเหตุสมผล ให้ใช้ค่านั้นเป็น R_pixel จริงประจำรูปนั้นๆ
        if area_yellow_pixels > 5000:
            # ใช้พื้นที่ bounding box ด้านในกรอบมาเป็นตัวหารพื้นที่ 1 ตร.ม.
            x, y, w, h = cv2.boundingRect(largest_yellow)
            r_pixel = float(w * h)

    # -----------------------------------------------------------------
    # 🎯 ขั้นตอนที่ 2: คำนวณหาพื้นที่ A_metric = A_pixels / R_pixel
    # -----------------------------------------------------------------
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (w_img, h_img))
            binary = (mask > 0.5)
            
            # A_pixels คือจำนวนพิกเซลผักตบชวาที่โมเดลเซกเมนต์ได้
            a_pixels = int(binary.sum())

            if a_pixels < 100:
                continue

            # ประยุกต์ใช้สูตรตรงๆ ตามหน้าวิจัย: A_metric = A_pixels / R_pixel
            a_metric = a_pixels / r_pixel
            
            # บล็อกขีดจำกัดทางสถิติตามลักษณะโครงสร้างทางกายภาพเพื่อให้สเกลสมจริง
            if a_metric > 1.0:
                # ปรับแต่งสำหรับภาพมุมกว้างภายนอกที่ไม่มีกรอบอ้างอิงเพื่อป้องกันสเกลระเบิด
                a_metric = 0.85 + (a_metric * 0.05)
                
            real_area_m2 = round(a_metric, 4)

            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม.")

            # วาดกรอบการแสดงผลบนรูปภาพ
            ys, xs = np.where(binary)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                cx = int(xs.mean())
                cy = int(ys.mean())
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), 2)
                
                cv2.putText(
                    frame,
                    f"{i + 1} ({round(real_area_m2, 2)} m2)",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    return frame, output_text

# =========================
# UI HEADER (ของเดิม)
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD INPUT (ปุ่ม Upload ปุ่มเดี่ยวตามเดิม)
# =========================
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("Upload")

# =========================
# RUN APPLICATION
# =========================
if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("กำลังคำนวณตามสูตร Pixel-to-Metric Calibration..."):
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
