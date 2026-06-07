import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import math

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Phak Top Chawa",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS (เวอร์ชันดีไซน์ของเน่ + ปรับแต่งส่วนสไลด์บาร์เพิ่มเติมให้เข้าธีมเขียว)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        180deg,
        #eef8ec 0%,
        #f8fff6 100
    ) !important;
}

/* TEXT COLOR */
.stMarkdown p, 
.stMarkdown span, 
.stText, 
.stSubheader, 
.stHeader,
h1, h2, h3 {
    color: #1b5e20 !important;
}

/* TITLE BANNER */
.main-title {
    background: linear-gradient(
        90deg,
        #1b5e20,
        #388e3c
    );
    color: white !important;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 16px;
}

/* SUB TITLE */
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 35px;
    font-weight: 500;
}

/* FILE UPLOADER DESIGN (ขอบเขียวประ) */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px dashed #1b5e20 !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

.stFileUploader * {
    color: #1b5e20 !important;
}

.stFileUploader button {
    border-radius: 12px !important;
    border: 1px solid #1b5e20 !important;
    background: white !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
}

/* BUTTON DESIGN */
.stButton button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #1b5e20,
        #388e3c
    );
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
    background: linear-gradient(
        90deg,
        #14461a,
        #2e7d32
    );
}

/* RESULT IMAGE */
img {
    border-radius: 20px;
    margin-top: 10px;
}

/* SIDEBAR DESIGN */
[data-testid="stSidebar"] {
    background-color: #f1f9f0 !important;
    border-right: 1px solid #c8e6c9 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR PARAMETERS (ส่วนเพิ่มขยายที่ทำเพิ่มเติมตามบรีฟอาจารย์)
# =========================
st.sidebar.markdown("### ⚙️ ปรับสเกลภาพถ่าย")

focal_length = st.sidebar.number_input(
    "Focal Length (mm):", 
    min_value=1.0, 
    max_value=500.0, 
    value=26.0, 
    step=1.0,
    help="ทางยาวโฟกัสของเลนส์กล้อง"
)

zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", 
    min_value=0.5, 
    max_value=50.0, 
    value=1.0, 
    step=0.1,
    help="ระยะซูมของภาพถ่าย"
)

# =========================
# โหลดโมเดล
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# ฟังก์ชันตรวจจับ (อัปเกรดระบบคำนวณพื้นที่เป็น ตร.ม. และขยายตัวเลขบนรูป)
# =========================
def detect(frame, f_length, zoom):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    
    # พารามิเตอร์อ้างอิงระนาบทัศนศาสตร์มุมกล้องหน้างาน
    d_field = 3.2
    theta_rad = math.radians(43.0)
    horizontal_dist = d_field * math.cos(theta_rad)
    
    # คำนวณอัตราส่วนการแปลงจาก Pixel เป็นตารางเมตรเชิงแสง (ตัวคูณคงสเกลเดิมที่พี่ใช้งาน)
    optical_scale = (f_length / 26.0) * zoom
    pixel_to_m2_ratio = 185000.0 * (optical_scale ** 1.2)

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (w_img, h_img))
            binary = (mask > 0.5)
            area_pixels = int(binary.sum())

            if area_pixels < 100:
                continue

            ys, xs = np.where(binary)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())
            
            # คำนวณ Perspective อิงตามตำแหน่งความลึกแกน Y ของภาพถ่าย
            normalized_y = cy / h_img
            calculated_area = area_pixels / pixel_to_m2_ratio
            depth_multiplier = (1.0 / (normalized_y + 0.18)) * (horizontal_dist / 1.5)
            real_area_m2 = calculated_area * depth_multiplier

            # จัดการตัวคูณขั้นต่ำตามระยะความลึกให้เสถียร
            if normalized_y > 0.70:
                real_area_m2 = max(0.12, real_area_m2 * 0.85)
            else:
                real_area_m2 = max(0.20, real_area_m2)

            real_area_m2 = round(real_area_m2, 2)

            # 📋 แก้ไขรายงานผลหลักเป็นหน่วย ตร.ม. พร้อมพิกเซลกำกับตามงานที่ทำเพิ่มเติม
            output_text.append(f"กอ#{i+1}  {real_area_m2} ตร.ม. (ตำแหน่ง X:{cx}, Y:{cy})")

            contours, _ = cv2.findContours(
                (binary * 255).astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
            
            # 🖼️ แสดงเฉพาะ "ตัวเลขหมายเลขกอ" ขนาดใหญ่สะใจ ไม่ติด ID หรือขนาด ตร.ม. ซ้ำซ้อน
            cv2.putText(
                frame,
                f"{i + 1}",
                (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
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
# UPLOAD
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

        # ส่งตัวแปรสเกลฝั่งซ้ายเข้าไปประมวลผลเพิ่ม
        result_frame, texts = detect(frame, focal_length, zoom_factor)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # ผลลัพธ์ข้อความ
        st.subheader("📋 ผลการตรวจจับ")
        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวา")

        st.markdown("<br>", unsafe_allow_html=True)

        # ภาพผลลัพธ์
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
